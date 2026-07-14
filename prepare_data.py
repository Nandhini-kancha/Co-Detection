"""
Prepare real datasets into unified format for training.
Reads downloaded Kaggle datasets and creates unified_labels.csv

Usage: python prepare_data.py --data_dir ./data
"""

import os
import csv
import argparse
from PIL import Image


def prepare_pneumonia_dataset(data_dir):
    """Prepare the Paul Mooney Chest X-ray Pneumonia dataset."""
    base = os.path.join(data_dir, "pneumonia", "chest_xray")
    
    if not os.path.exists(base):
        print("  Pneumonia dataset not found")
        return []
    
    records = []
    
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(base, split)
        if not os.path.exists(split_dir):
            continue
        
        for cls_name in ['NORMAL', 'PNEUMONIA']:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            
            for img_name in os.listdir(cls_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(cls_dir, img_name)
                is_pneumonia = 1 if cls_name == 'PNEUMONIA' else 0
                
                records.append({
                    'image_path': os.path.abspath(img_path),
                    'pneumonia': is_pneumonia,
                    'tb': 0,
                    'normal': 1 if is_pneumonia == 0 else 0,
                    'source': 'kaggle_pneumonia'
                })
    
    print(f"  Pneumonia dataset: {len(records)} images")
    pn = sum(1 for r in records if r['pneumonia'] == 1)
    nm = sum(1 for r in records if r['normal'] == 1)
    print(f"    Pneumonia: {pn}, Normal: {nm}")
    return records


def prepare_shenzhen_dataset(data_dir):
    """Prepare the Shenzhen TB dataset."""
    # Handle nested directory: images/images/ from Kaggle extraction
    images_dir = os.path.join(data_dir, "shenzhen", "images", "images")
    if not os.path.exists(images_dir):
        images_dir = os.path.join(data_dir, "shenzhen", "images")
    
    metadata_csv = os.path.join(data_dir, "shenzhen", "shenzhen_metadata.csv")
    
    if not os.path.exists(images_dir):
        print("  Shenzhen TB dataset not found")
        return []
    
    # Try using metadata CSV if available
    tb_labels = {}
    if os.path.exists(metadata_csv):
        try:
            with open(metadata_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('study_id', row.get('filename', ''))
                    finding = row.get('findings', row.get('label', ''))
                    if filename:
                        is_tb = 1 if 'normal' not in str(finding).lower() else 0
                        tb_labels[filename] = is_tb
        except Exception:
            pass
    
    records = []
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img_path = os.path.join(images_dir, img_name)
        
        # Use metadata if available, otherwise use filename convention
        if img_name in tb_labels:
            is_tb = tb_labels[img_name]
        else:
            # Shenzhen naming: CHNCXR_NNNN_1.png = TB, CHNCXR_NNNN_0.png = Normal
            is_tb = 1 if '_1.' in img_name or '_1_' in img_name else 0
        
        records.append({
            'image_path': os.path.abspath(img_path),
            'pneumonia': 0,
            'tb': is_tb,
            'normal': 1 if is_tb == 0 else 0,
            'source': 'shenzhen'
        })
    
    print(f"  Shenzhen TB dataset: {len(records)} images")
    tb = sum(1 for r in records if r['tb'] == 1)
    nm = sum(1 for r in records if r['normal'] == 1)
    print(f"    TB: {tb}, Normal: {nm}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    
    print("Preparing datasets...")
    print()
    
    all_records = []
    
    # Pneumonia
    pn_records = prepare_pneumonia_dataset(args.data_dir)
    all_records.extend(pn_records)
    
    # Shenzhen TB
    tb_records = prepare_shenzhen_dataset(args.data_dir)
    all_records.extend(tb_records)
    
    if not all_records:
        print("\nNo datasets found!")
        return
    
    # Verify images exist
    valid_records = []
    missing = 0
    for r in all_records:
        if os.path.exists(r['image_path']):
            valid_records.append(r)
        else:
            missing += 1
    
    if missing > 0:
        print(f"\n  Warning: {missing} images not found, skipped")
    
    # Save unified CSV
    unified_path = os.path.join(args.data_dir, "unified_labels.csv")
    with open(unified_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'pneumonia', 'tb', 'normal', 'source'])
        writer.writeheader()
        writer.writerows(valid_records)
    
    total_pn = sum(1 for r in valid_records if r['pneumonia'] == 1)
    total_tb = sum(1 for r in valid_records if r['tb'] == 1)
    total_nm = sum(1 for r in valid_records if r['normal'] == 1)
    
    print(f"\n{'='*60}")
    print(f"UNIFIED DATASET READY")
    print(f"{'='*60}")
    print(f"  Total images: {len(valid_records)}")
    print(f"    Pneumonia:  {total_pn}")
    print(f"    TB:         {total_tb}")
    print(f"    Normal:     {total_nm}")
    print(f"  Saved to: {unified_path}")
    print(f"\n  Train with:")
    print(f"  python train.py --data_dir {args.data_dir} --epochs 20 --batch_size 16 --num_workers 0")


if __name__ == '__main__':
    main()
