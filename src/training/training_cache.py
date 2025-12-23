#!/usr/bin/env python
"""
Training Data Caching System
Efficient caching for graph correction training data to speed up iterations
"""

import os
import json
import pickle
import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import gzip
import tempfile

import numpy as np


@dataclass
class CacheConfig:
    """Configuration for training data cache"""
    cache_dir: Path = Path("training_cache")
    max_cache_size_gb: float = 10.0  # Maximum cache size in GB
    compress_data: bool = True  # Use gzip compression
    cache_validation: bool = True  # Validate cache integrity
    auto_cleanup: bool = True  # Automatic cleanup of old entries
    max_age_days: int = 30  # Maximum age before cleanup
    preload_cache: bool = False  # Preload cache into memory
    cache_statistics: bool = True  # Track cache statistics


@dataclass
class CacheEntry:
    """Individual cache entry metadata"""
    cache_key: str
    file_path: Path
    creation_time: datetime
    generation_params: Dict
    data_size: int
    compression_ratio: float
    access_count: int = 0
    last_access: Optional[datetime] = None
    is_valid: bool = True


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    total_entries: int = 0
    total_size_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    generation_time_saved: float = 0.0
    average_compression_ratio: float = 0.0
    last_cleanup: Optional[datetime] = None


class TrainingDataCache:
    """
    High-performance caching system for training data
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup cache directory
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.stats_file = self.cache_dir / "cache_statistics.json"
        
        # In-memory structures
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.cache_statistics = CacheStatistics()
        self.memory_cache: Dict[str, Any] = {}  # For preloaded data
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing cache metadata
        self._load_metadata()
        self._load_statistics()
        
        # Perform initial cleanup if enabled
        if config.auto_cleanup:
            self._cleanup_old_entries()
        
        self.logger.info(f"Training cache initialized: {len(self.cache_entries)} entries, "
                        f"{self._get_total_size_mb():.1f} MB")
    
    def generate_cache_key(self, generation_params: Dict) -> str:
        """Generate deterministic cache key from generation parameters"""
        # Create a normalized parameter dict for hashing
        normalized_params = self._normalize_params(generation_params)
        
        # Convert to sorted JSON string for consistent hashing
        param_string = json.dumps(normalized_params, sort_keys=True)
        
        # Generate SHA-256 hash
        cache_key = hashlib.sha256(param_string.encode()).hexdigest()[:16]
        
        return cache_key
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if data is available in cache"""
        with self._lock:
            if cache_key not in self.cache_entries:
                return False
            
            entry = self.cache_entries[cache_key]
            
            # Validate cache entry
            if not entry.file_path.exists():
                self.logger.warning(f"Cache file missing: {entry.file_path}")
                self._invalidate_entry(cache_key)
                return False
            
            return entry.is_valid
    
    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache"""
        start_time = time.time()
        
        with self._lock:
            if not self.is_cached(cache_key):
                self.cache_statistics.cache_misses += 1
                return None
            
            entry = self.cache_entries[cache_key]
            
            # Check memory cache first
            if self.config.preload_cache and cache_key in self.memory_cache:
                self.logger.debug(f"Memory cache hit: {cache_key}")
                self._update_access_stats(entry)
                self.cache_statistics.cache_hits += 1
                return self.memory_cache[cache_key]
            
            # Load from disk
            try:
                data = self._load_data_from_disk(entry.file_path)
                
                # Store in memory cache if preloading is enabled
                if self.config.preload_cache:
                    self.memory_cache[cache_key] = data
                
                self._update_access_stats(entry)
                self.cache_statistics.cache_hits += 1
                
                load_time = time.time() - start_time
                self.logger.debug(f"Cache hit: {cache_key} (loaded in {load_time:.3f}s)")
                
                return data
                
            except Exception as e:
                self.logger.error(f"Failed to load cached data {cache_key}: {e}")
                self._invalidate_entry(cache_key)
                self.cache_statistics.cache_misses += 1
                return None
    
    def cache_data(self, cache_key: str, data: Any, generation_params: Dict,
                   generation_time: float = 0.0) -> bool:
        """Store data in cache"""
        start_time = time.time()
        
        with self._lock:
            try:
                # Generate cache file path
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if self.config.compress_data:
                    cache_file = cache_file.with_suffix('.pkl.gz')
                
                # Save data to disk
                original_size = self._estimate_data_size(data)
                compressed_size = self._save_data_to_disk(data, cache_file)
                compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
                
                # Create cache entry
                entry = CacheEntry(
                    cache_key=cache_key,
                    file_path=cache_file,
                    creation_time=datetime.now(),
                    generation_params=generation_params.copy(),
                    data_size=compressed_size,
                    compression_ratio=compression_ratio
                )
                
                self.cache_entries[cache_key] = entry
                
                # Store in memory cache if preloading is enabled
                if self.config.preload_cache:
                    self.memory_cache[cache_key] = data
                
                # Update statistics
                self.cache_statistics.total_entries = len(self.cache_entries)
                self.cache_statistics.total_size_bytes = sum(
                    e.data_size for e in self.cache_entries.values()
                )
                self.cache_statistics.generation_time_saved += generation_time
                
                cache_time = time.time() - start_time
                self.logger.info(f"Cached data: {cache_key} "
                               f"({compressed_size/1024/1024:.1f} MB, "
                               f"compression: {compression_ratio:.2f}, "
                               f"cached in {cache_time:.3f}s)")
                
                # Check cache size limits
                self._enforce_cache_limits()
                
                # Save metadata periodically
                if len(self.cache_entries) % 10 == 0:
                    self._save_metadata()
                    self._save_statistics()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache data {cache_key}: {e}")
                return False
    
    def warm_cache(self, cache_keys: List[str]) -> int:
        """Preload specified cache entries into memory"""
        if not self.config.preload_cache:
            self.logger.warning("Cache preloading is disabled")
            return 0
        
        loaded_count = 0
        start_time = time.time()
        
        self.logger.info(f"Warming cache for {len(cache_keys)} entries...")
        
        for cache_key in cache_keys:
            if cache_key in self.memory_cache:
                continue
                
            data = self.get_cached_data(cache_key)
            if data is not None:
                loaded_count += 1
        
        warm_time = time.time() - start_time
        self.logger.info(f"Cache warmed: {loaded_count}/{len(cache_keys)} entries "
                        f"in {warm_time:.2f}s")
        
        return loaded_count
    
    def get_cache_info(self) -> Dict:
        """Get comprehensive cache information"""
        with self._lock:
            total_size_mb = self._get_total_size_mb()
            memory_size_mb = self._get_memory_cache_size_mb()
            
            hit_rate = (self.cache_statistics.cache_hits / 
                       max(1, self.cache_statistics.cache_hits + self.cache_statistics.cache_misses))
            
            return {
                'total_entries': len(self.cache_entries),
                'total_size_mb': total_size_mb,
                'memory_cache_size_mb': memory_size_mb,
                'cache_hit_rate': hit_rate,
                'cache_hits': self.cache_statistics.cache_hits,
                'cache_misses': self.cache_statistics.cache_misses,
                'generation_time_saved_hours': self.cache_statistics.generation_time_saved / 3600,
                'average_compression_ratio': self._get_average_compression_ratio(),
                'oldest_entry': self._get_oldest_entry_age(),
                'most_accessed': self._get_most_accessed_entries(5)
            }
    
    def cleanup_cache(self, force: bool = False) -> Dict:
        """Clean up cache entries based on age and usage"""
        with self._lock:
            cleanup_stats = {
                'entries_removed': 0,
                'size_freed_mb': 0.0,
                'invalid_entries_removed': 0
            }
            
            current_time = datetime.now()
            entries_to_remove = []
            
            for cache_key, entry in self.cache_entries.items():
                should_remove = False
                
                # Remove invalid entries
                if not entry.is_valid or not entry.file_path.exists():
                    should_remove = True
                    cleanup_stats['invalid_entries_removed'] += 1
                
                # Remove old entries
                elif self.config.max_age_days > 0:
                    age_days = (current_time - entry.creation_time).days
                    if age_days > self.config.max_age_days:
                        should_remove = True
                
                # Force cleanup if requested
                elif force:
                    should_remove = True
                
                if should_remove:
                    entries_to_remove.append(cache_key)
                    cleanup_stats['size_freed_mb'] += entry.data_size / 1024 / 1024
            
            # Remove entries
            for cache_key in entries_to_remove:
                self._remove_cache_entry(cache_key)
                cleanup_stats['entries_removed'] += 1
            
            # Update statistics
            self.cache_statistics.last_cleanup = current_time
            self._save_metadata()
            self._save_statistics()
            
            self.logger.info(f"Cache cleanup completed: "
                           f"removed {cleanup_stats['entries_removed']} entries, "
                           f"freed {cleanup_stats['size_freed_mb']:.1f} MB")
            
            return cleanup_stats
    
    def clear_cache(self) -> bool:
        """Clear all cache data"""
        with self._lock:
            try:
                # Remove all cache files
                for entry in self.cache_entries.values():
                    if entry.file_path.exists():
                        entry.file_path.unlink()
                
                # Clear in-memory structures
                self.cache_entries.clear()
                self.memory_cache.clear()
                self.cache_statistics = CacheStatistics()
                
                # Remove metadata files
                if self.metadata_file.exists():
                    self.metadata_file.unlink()
                if self.stats_file.exists():
                    self.stats_file.unlink()
                
                self.logger.info("Cache cleared completely")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to clear cache: {e}")
                return False
    
    def _normalize_params(self, params: Dict) -> Dict:
        """Normalize parameters for consistent hashing"""
        normalized = {}
        
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                normalized[key] = tuple(value)
            elif isinstance(value, dict):
                normalized[key] = self._normalize_params(value)
            elif isinstance(value, Path):
                normalized[key] = str(value)
            elif isinstance(value, np.ndarray):
                normalized[key] = value.tolist()
            else:
                normalized[key] = value
        
        return normalized
    
    def _load_data_from_disk(self, file_path: Path) -> Any:
        """Load data from disk with optional decompression"""
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def _save_data_to_disk(self, data: Any, file_path: Path) -> int:
        """Save data to disk with optional compression"""
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use temporary file for atomic writes in the same directory as destination
        with tempfile.NamedTemporaryFile(delete=False, dir=file_path.parent) as tmp_file:
            temp_path = Path(tmp_file.name)
        
        try:
            if file_path.suffix == '.gz':
                with gzip.open(temp_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic move (now guaranteed to be on same filesystem)
            temp_path.replace(file_path)
            
            return file_path.stat().st_size
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate data size in bytes"""
        try:
            # Quick estimate using pickle
            return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            return 0
    
    def _update_access_stats(self, entry: CacheEntry):
        """Update access statistics for cache entry"""
        entry.access_count += 1
        entry.last_access = datetime.now()
    
    def _invalidate_entry(self, cache_key: str):
        """Mark cache entry as invalid"""
        if cache_key in self.cache_entries:
            self.cache_entries[cache_key].is_valid = False
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry completely"""
        if cache_key in self.cache_entries:
            entry = self.cache_entries[cache_key]
            if entry.file_path.exists():
                entry.file_path.unlink()
            del self.cache_entries[cache_key]
        
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
    
    def _enforce_cache_limits(self):
        """Enforce cache size limits"""
        total_size_gb = self._get_total_size_mb() / 1024
        
        if total_size_gb > self.config.max_cache_size_gb:
            # Remove least recently used entries
            entries_by_access = sorted(
                self.cache_entries.items(),
                key=lambda x: (x[1].last_access or x[1].creation_time)
            )
            
            size_to_remove = (total_size_gb - self.config.max_cache_size_gb) * 1024 * 1024 * 1024
            size_removed = 0
            
            for cache_key, entry in entries_by_access:
                if size_removed >= size_to_remove:
                    break
                
                size_removed += entry.data_size
                self._remove_cache_entry(cache_key)
                self.logger.debug(f"Removed cache entry for size limit: {cache_key}")
    
    def _cleanup_old_entries(self):
        """Remove old cache entries"""
        if self.config.max_age_days <= 0:
            return
        
        current_time = datetime.now()
        old_entries = []
        
        for cache_key, entry in self.cache_entries.items():
            age_days = (current_time - entry.creation_time).days
            if age_days > self.config.max_age_days:
                old_entries.append(cache_key)
        
        for cache_key in old_entries:
            self._remove_cache_entry(cache_key)
        
        if old_entries:
            self.logger.info(f"Cleaned up {len(old_entries)} old cache entries")
    
    def _get_total_size_mb(self) -> float:
        """Get total cache size in MB"""
        return sum(entry.data_size for entry in self.cache_entries.values()) / 1024 / 1024
    
    def _get_memory_cache_size_mb(self) -> float:
        """Get memory cache size in MB"""
        total_size = 0
        for data in self.memory_cache.values():
            total_size += self._estimate_data_size(data)
        return total_size / 1024 / 1024
    
    def _get_average_compression_ratio(self) -> float:
        """Get average compression ratio"""
        if not self.cache_entries:
            return 1.0
        
        total_ratio = sum(entry.compression_ratio for entry in self.cache_entries.values())
        return total_ratio / len(self.cache_entries)
    
    def _get_oldest_entry_age(self) -> Optional[int]:
        """Get age of oldest entry in days"""
        if not self.cache_entries:
            return None
        
        oldest_time = min(entry.creation_time for entry in self.cache_entries.values())
        return (datetime.now() - oldest_time).days
    
    def _get_most_accessed_entries(self, count: int) -> List[Dict]:
        """Get most accessed cache entries"""
        sorted_entries = sorted(
            self.cache_entries.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        
        return [
            {
                'cache_key': cache_key[:8] + '...',
                'access_count': entry.access_count,
                'size_mb': entry.data_size / 1024 / 1024
            }
            for cache_key, entry in sorted_entries[:count]
        ]
    
    def _load_metadata(self):
        """Load cache metadata from disk"""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for entry_data in metadata.get('entries', []):
                entry = CacheEntry(
                    cache_key=entry_data['cache_key'],
                    file_path=Path(entry_data['file_path']),
                    creation_time=datetime.fromisoformat(entry_data['creation_time']),
                    generation_params=entry_data['generation_params'],
                    data_size=entry_data['data_size'],
                    compression_ratio=entry_data['compression_ratio'],
                    access_count=entry_data.get('access_count', 0),
                    last_access=datetime.fromisoformat(entry_data['last_access']) 
                                if entry_data.get('last_access') else None,
                    is_valid=entry_data.get('is_valid', True)
                )
                self.cache_entries[entry.cache_key] = entry
            
            self.logger.debug(f"Loaded {len(self.cache_entries)} cache entries from metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to load cache metadata: {e}")
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            # Check if cache directory still exists
            if not self.cache_dir.exists():
                return  # Skip saving if directory was removed
                
            metadata = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'entries': []
            }
            
            for entry in self.cache_entries.values():
                entry_data = {
                    'cache_key': entry.cache_key,
                    'file_path': str(entry.file_path),
                    'creation_time': entry.creation_time.isoformat(),
                    'generation_params': entry.generation_params,
                    'data_size': entry.data_size,
                    'compression_ratio': entry.compression_ratio,
                    'access_count': entry.access_count,
                    'last_access': entry.last_access.isoformat() if entry.last_access else None,
                    'is_valid': entry.is_valid
                }
                metadata['entries'].append(entry_data)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            # Only log error if it's not a "directory doesn't exist" issue
            if "No such file or directory" not in str(e):
                self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _load_statistics(self):
        """Load cache statistics from disk"""
        if not self.stats_file.exists():
            return
        
        try:
            with open(self.stats_file, 'r') as f:
                stats_data = json.load(f)
            
            self.cache_statistics = CacheStatistics(
                total_entries=stats_data.get('total_entries', 0),
                total_size_bytes=stats_data.get('total_size_bytes', 0),
                cache_hits=stats_data.get('cache_hits', 0),
                cache_misses=stats_data.get('cache_misses', 0),
                generation_time_saved=stats_data.get('generation_time_saved', 0.0),
                average_compression_ratio=stats_data.get('average_compression_ratio', 0.0),
                last_cleanup=datetime.fromisoformat(stats_data['last_cleanup']) 
                            if stats_data.get('last_cleanup') else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load cache statistics: {e}")
    
    def _save_statistics(self):
        """Save cache statistics to disk"""
        try:
            # Check if cache directory still exists
            if not self.cache_dir.exists():
                return  # Skip saving if directory was removed
                
            stats_data = asdict(self.cache_statistics)
            if stats_data['last_cleanup']:
                stats_data['last_cleanup'] = self.cache_statistics.last_cleanup.isoformat()
            
            with open(self.stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
            
        except Exception as e:
            # Only log error if it's not a "directory doesn't exist" issue
            if "No such file or directory" not in str(e):
                self.logger.error(f"Failed to save cache statistics: {e}")
    
    def __del__(self):
        """Cleanup when cache object is destroyed"""
        try:
            self._save_metadata()
            self._save_statistics()
        except:
            pass