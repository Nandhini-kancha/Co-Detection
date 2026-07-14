"""
Download REAL chest X-ray images for Pneumonia & TB training.
Uses huggingface_hub to download directly from dataset repositories.

Usage:
    python download_real_data.py
    python download_real_data.py --max_samples 500
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from PIL import Image
from io import BytesIO


def download_pneumonia_dataset(data_dir, max_samples=None):
    """Download pneumonia chest X-ray dataset using huggingface_hub."""
    from huggingface_hub import HfApi, hf_hub_download
    
    print("=" * 60)
    print("DOWNLOADING PNEUMONIA CHEST X-RAY DATASET")
    print("=" * 60)
    
    images_dir = os.path.join(data_dir, "pneumonia", "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Try multiple dataset sources
    dataset_candidates = [
        "trpakov/chest-xray-classification",
        "hf-vision/chest-xray-pneumonia",
        "benschill/chest-xray-pneumonia",
    ]
    
    api = HfApi()
    
    for dataset_id in dataset_candidates:
        try:
            print(f"  Trying: {dataset_id}")
            
            # List files in the repo
            files = api.list_repo_tree(dataset_id, repo_type="dataset", recursive=True)
            image_files = []
            
            for f in files:
                path = f.rfilename if hasattr(f, 'rfilename') else str(f)
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(path)
            
            if not image_files:
                print(f"    No images found, skipping...")
                continue
            
            print(f"    Found {len(image_files)} images")
            
            records = []
            count = 0
            limit = max_samples or len(image_files)
            
            for img_path in image_files:
                if count >= limit:
                    break
                
                # Determine label from path
                path_lower = img_path.lower()
                if 'pneumonia' in path_lower:
                    is_pneumonia = 1
                elif 'normal' in path_lower:
                    is_pneumonia = 0
                else:
                    continue  # Skip unknown
                
                try:
                    # Download the file
                    local_path = hf_hub_download(
                        repo_id=dataset_id,
                        filename=img_path,
                        repo_type="dataset"
                    )
                    
                    # Copy to our directory
                    img = Image.open(local_path).convert('RGB')
                    filename = f"pn_{count:05d}.png"
                    dest = os.path.join(images_dir, filename)
                    img.save(dest)
                    
                    records.append({
                        'filename': filename,
                        'pneumonia': is_pneumonia,
                        'tb': 0,
                        'normal': 1 if is_pneumonia == 0 else 0
                    })
                    
                    count += 1
                    if count % 50 == 0:
                        print(f"    Downloaded {count}/{limit} images...")
                        
                except Exception as e:
                    continue
            
            if records:
                # Save labels
                labels_path = os.path.join(data_dir, "pneumonia", "labels.csv")
                with open(labels_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'pneumonia', 'tb', 'normal'])
                    writer.writeheader()
                    writer.writerows(records)
                
                pn_count = sum(1 for r in records if r['pneumonia'] == 1)
                nm_count = sum(1 for r in records if r['normal'] == 1)
                print(f"\n  SUCCESS! {count} images downloaded:")
                print(f"    Pneumonia: {pn_count}")
                print(f"    Normal: {nm_count}")
                return True
                
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    print("  All sources failed for pneumonia dataset.")
    print("  Trying Kaggle download...")
    return download_via_kaggle(data_dir, "pneumonia", max_samples)


def download_tb_dataset(data_dir, max_samples=None):
    """Download TB chest X-ray dataset using huggingface_hub."""
    from huggingface_hub import HfApi, hf_hub_download
    
    print("=" * 60)
    print("DOWNLOADING TUBERCULOSIS CHEST X-RAY DATASET")
    print("=" * 60)
    
    images_dir = os.path.join(data_dir, "tb_hf", "images")
    os.makedirs(images_dir, exist_ok=True)
    
    dataset_candidates = [
        "alkzar90/TB_Chest_Radiography_Database",
        "MohammedAlQura662/TB_Chest_Radiography_Database",
    ]
    
    api = HfApi()
    
    for dataset_id in dataset_candidates:
        try:
            print(f"  Trying: {dataset_id}")
            
            files = api.list_repo_tree(dataset_id, repo_type="dataset", recursive=True)
            image_files = []
            
            for f in files:
                path = f.rfilename if hasattr(f, 'rfilename') else str(f)
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(path)
            
            if not image_files:
                print(f"    No images found, skipping...")
                continue
            
            print(f"    Found {len(image_files)} images")
            
            records = []
            count = 0
            limit = max_samples or len(image_files)
            
            for img_path in image_files:
                if count >= limit:
                    break
                
                path_lower = img_path.lower()
                if 'tuberculosis' in path_lower or '/tb/' in path_lower or '_tb' in path_lower:
                    is_tb = 1
                elif 'normal' in path_lower or 'healthy' in path_lower:
                    is_tb = 0
                else:
                    continue
                
                try:
                    local_path = hf_hub_download(
                        repo_id=dataset_id,
                        filename=img_path,
                        repo_type="dataset"
                    )
                    
                    img = Image.open(local_path).convert('RGB')
                    filename = f"tb_{count:05d}.png"
                    dest = os.path.join(images_dir, filename)
                    img.save(dest)
                    
                    records.append({
                        'filename': filename,
                        'pneumonia': 0,
                        'tb': is_tb,
                        'normal': 1 if is_tb == 0 else 0
                    })
                    
                    count += 1
                    if count % 50 == 0:
                        print(f"    Downloaded {count}/{limit} images...")
                        
                except Exception as e:
                    continue
            
            if records:
                labels_path = os.path.join(data_dir, "tb_hf", "labels.csv")
                with open(labels_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'pneumonia', 'tb', 'normal'])
                    writer.writeheader()
                    writer.writerows(records)
                
                tb_count = sum(1 for r in records if r['tb'] == 1)
                nm_count = sum(1 for r in records if r['normal'] == 1)
                print(f"\n  SUCCESS! {count} images downloaded:")
                print(f"    TB: {tb_count}")
                print(f"    Normal: {nm_count}")
                return True
                
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    print("  All HF sources failed for TB dataset.")
    return download_via_kaggle(data_dir, "tb", max_samples)


def download_via_kaggle(data_dir, dataset_type, max_samples):
    """Fallback: try downloading via Kaggle CLI."""
    try:
        import subprocess
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, Exception):
        print("\n  Kaggle CLI not available.")
        print("  To install: pip install kaggle")
        print("  Then set up credentials: https://www.kaggle.com/docs/api")
        if dataset_type == "pneumonia":
            print("  Download manually: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        else:
            print("  Download manually: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
        return False
    
    return False


def create_unified_csv(data_dir):
    """Merge all downloaded dataset labels into one unified CSV."""
    print("\n" + "=" * 60)
    print("CREATING UNIFIED TRAINING DATASET")
    print("=" * 60)
    
    all_records = []
    
    for subdir, source_name in [("pneumonia", "pneumonia_hf"), ("tb_hf", "tb_hf"), ("shenzhen", "shenzhen")]:
        labels_path = os.path.join(data_dir, subdir, "labels.csv")
        if os.path.exists(labels_path):
            images_dir = os.path.join(data_dir, subdir, "images")
            with open(labels_path, 'r') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    img_path = os.path.join(images_dir, row['filename'])
                    if os.path.exists(img_path):
                        all_records.append({
                            'image_path': img_path,
                            'filename': row['filename'],
                            'pneumonia': int(row['pneumonia']),
                            'tb': int(row['tb']),
                            'normal': int(row['normal']),
                            'source': source_name
                        })
                        count += 1
            if count > 0:
                print(f"  {source_name}: {count} images")
    
    if not all_records:
        print("  No datasets found!")
        return False
    
    unified_path = os.path.join(data_dir, "unified_labels.csv")
    with open(unified_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'filename', 'pneumonia', 'tb', 'normal', 'source'])
        writer.writeheader()
        writer.writerows(all_records)
    
    total_pn = sum(1 for r in all_records if r['pneumonia'] == 1)
    total_tb = sum(1 for r in all_records if r['tb'] == 1)
    total_nm = sum(1 for r in all_records if r['normal'] == 1)
    
    print(f"\n  Total: {len(all_records)} images")
    print(f"    Pneumonia: {total_pn}")
    print(f"    TB: {total_tb}")
    print(f"    Normal: {total_nm}")
    print(f"  Saved to: {unified_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Download real chest X-ray datasets')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    print("\nReal Chest X-Ray Dataset Downloader")
    print("=" * 60)
    
    download_pneumonia_dataset(args.data_dir, args.max_samples)
    print()
    download_tb_dataset(args.data_dir, args.max_samples)
    print()
    create_unified_csv(args.data_dir)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS - Train on real data:")
    print(f"  python train.py --data_dir {args.data_dir} --epochs 20 --batch_size 16 --num_workers 0")
    print("=" * 60)


if __name__ == '__main__':
    main()
