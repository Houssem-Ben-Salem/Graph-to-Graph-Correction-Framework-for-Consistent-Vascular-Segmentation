#!/usr/bin/env python
"""Validate dataset and count complete cases"""

import os
import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def validate_dataset(dataset_path):
    """Validate dataset and return list of complete cases"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        return []
    
    patient_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories")
    
    complete_cases = []
    incomplete_cases = []
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        
        # Check for image file
        image_path = patient_dir / 'image' / f'{patient_id}.nii.gz'
        # Check for label file
        label_path = patient_dir / 'label' / f'{patient_id}.nii.gz'
        
        # Check what exists
        has_image = image_path.exists()
        has_label = label_path.exists()
        has_image_dir = (patient_dir / 'image').exists()
        has_label_dir = (patient_dir / 'label').exists()
        
        if has_image and has_label:
            # Check file sizes to ensure they're not empty
            image_size = image_path.stat().st_size if has_image else 0
            label_size = label_path.stat().st_size if has_label else 0
            
            if image_size > 1000 and label_size > 1000:  # At least 1KB each
                complete_cases.append(patient_id)
            else:
                incomplete_cases.append({
                    'patient_id': patient_id,
                    'issue': f'Small files (image: {image_size}B, label: {label_size}B)'
                })
        else:
            issue_details = []
            if not has_image_dir:
                issue_details.append('no image directory')
            elif not has_image:
                issue_details.append('no image file')
            
            if not has_label_dir:
                issue_details.append('no label directory')
            elif not has_label:
                issue_details.append('no label file')
            
            incomplete_cases.append({
                'patient_id': patient_id,
                'issue': ', '.join(issue_details)
            })
    
    return complete_cases, incomplete_cases


def main():
    parser = argparse.ArgumentParser(description='Validate dataset')
    parser.add_argument('--dataset', type=str, default='DATASET/Parse_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--save-list', type=str,
                        help='Save list of complete cases to file')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information about incomplete cases')
    
    args = parser.parse_args()
    
    print("Dataset Validation")
    print("=" * 50)
    
    complete_cases, incomplete_cases = validate_dataset(args.dataset)
    
    print(f"\n✅ Complete cases: {len(complete_cases)}")
    print(f"❌ Incomplete cases: {len(incomplete_cases)}")
    print(f"📊 Total directories: {len(complete_cases) + len(incomplete_cases)}")
    print(f"📈 Completion rate: {len(complete_cases)/(len(complete_cases) + len(incomplete_cases))*100:.1f}%")
    
    if incomplete_cases and args.verbose:
        print(f"\nIncomplete cases:")
        for case in incomplete_cases[:10]:  # Show first 10
            print(f"  - {case['patient_id']}: {case['issue']}")
        if len(incomplete_cases) > 10:
            print(f"  ... and {len(incomplete_cases) - 10} more")
    
    if args.save_list:
        with open(args.save_list, 'w') as f:
            for case_id in complete_cases:
                f.write(f"{case_id}\n")
        print(f"\n💾 Complete cases list saved to: {args.save_list}")
    
    # Show some example complete cases
    if complete_cases:
        print(f"\nFirst 10 complete cases:")
        for case_id in complete_cases[:10]:
            print(f"  ✅ {case_id}")
        if len(complete_cases) > 10:
            print(f"  ... and {len(complete_cases) - 10} more")
    
    # Check PA000080 specifically since it was mentioned in the error
    pa000080_path = Path(args.dataset) / "PA000080"
    if pa000080_path.exists():
        image_exists = (pa000080_path / "image" / "PA000080.nii.gz").exists()
        label_exists = (pa000080_path / "label" / "PA000080.nii.gz").exists()
        print(f"\n🔍 PA000080 check:")
        print(f"  Directory exists: ✅")
        print(f"  Image file exists: {'✅' if image_exists else '❌'}")
        print(f"  Label file exists: {'✅' if label_exists else '❌'}")
    
    return len(complete_cases)


if __name__ == '__main__':
    complete_count = main()
    
    if complete_count == 0:
        print("\n❌ No complete cases found!")
        sys.exit(1)
    elif complete_count < 10:
        print(f"\n⚠️  Warning: Only {complete_count} complete cases found. This might not be enough for robust training.")
    else:
        print(f"\n✅ Ready for training with {complete_count} complete cases!")