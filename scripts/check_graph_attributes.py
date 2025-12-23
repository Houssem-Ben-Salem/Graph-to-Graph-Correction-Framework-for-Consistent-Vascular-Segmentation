#!/usr/bin/env python
"""Check what attributes are available in the graphs"""

import sys
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🔍 Checking Graph Attributes\n")
    
    # Load one example
    patient_id = "PA000005"
    gt_file = Path(f"extracted_graphs/{patient_id}/{patient_id}_GT.pkl")
    
    with open(gt_file, 'rb') as f:
        gt_graph = pickle.load(f)
    
    print(f"GT Graph Type: {type(gt_graph)}")
    print(f"GT Graph Attributes: {dir(gt_graph)}")
    
    # Check specific attributes
    attrs_to_check = [
        'nodes', 'edges', 'node_positions', 'node_attributes', 
        'edge_attributes', 'positions', 'coordinates', 'spatial_data',
        'node_data', 'edge_data', 'graph'
    ]
    
    print(f"\nAttribute Check:")
    for attr in attrs_to_check:
        if hasattr(gt_graph, attr):
            value = getattr(gt_graph, attr)
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"  ✅ {attr}: {type(value)} (length: {len(value)})")
                # Sample first item if it's a dict
                if isinstance(value, dict) and len(value) > 0:
                    first_key = list(value.keys())[0]
                    print(f"      Example: {first_key} -> {value[first_key]}")
            else:
                print(f"  ✅ {attr}: {type(value)} = {value}")
        else:
            print(f"  ❌ {attr}: Not found")

if __name__ == "__main__":
    main()