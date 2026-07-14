"""
Download the 3 required datasets for Chest X-ray Co-Detection:
1. NIH ChestX-ray14 (Pneumonia labels)
2. Shenzhen TB Dataset
3. RSNA Pneumonia Detection Dataset

Prerequisites:
    1. Create Kaggle account at https://www.kaggle.com
    2. Go to Settings > API > Create New Token
    3. Place kaggle.json at C:\\Users\\<user>\\.kaggle\\kaggle.json

Usage:
    python download_datasets.py                    # Download all 3
    python download_datasets.py --dataset nih      # Only NIH
    python download_datasets.py --dataset shenzhen # Only Shenzhen TB
    python download_datasets.py --dataset rsna     # Only RSNA
    python download_datasets.py --small            # Download small subsets
"""

import os
import sys
import argparse
import shutil
import csv
import random
from pathlib import Path


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_json):
        print("[OK] Kaggle credentials found")
        return True
    
    # Also check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("[OK] Kaggle credentials found (env vars)")
        return True
    
    print("[ERROR] Kaggle credentials not found!")
    print()
    print("  Setup instructions:")
    print("  1. Go to https://www.kaggle.com and sign up (free)")
    print("  2. Click profile icon > Settings")
    print("  3. Scroll to API section > Create New Token")
    print("  4. Move downloaded kaggle.json to:")
    print(f"     {kaggle_json}")
    print()
    return False


def download_nih_chestxray14(data_dir, small=False):
    """Download NIH ChestX-ray14 dataset from Kaggle."""
    print("=" * 60)
    print("1. NIH CHESTX-RAY14 DATASET")
    print("=" * 60)
    print("   112,120 frontal chest X-rays, 14 disease labels")
    print("   Size: ~45 GB (full) / ~1 GB sample")
    print()
    
    output_dir = os.path.join(data_dir, "nih")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if already downloaded
    csv_path = os.path.join(output_dir, "Data_Entry_2017.csv")
    images_dir = os.path.join(output_dir, "images")
    
    if os.path.exists(csv_path) and os.path.exists(images_dir):
        img_count = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
        if img_count > 100:
            print(f"   Already downloaded ({img_count} images). Skipping.")
            return True
    
    try:
        import opendatasets as od
        
        if small:
            # Download sample dataset (smaller)
            print("   Downloading NIH sample dataset...")
            od.download(
                "https://www.kaggle.com/datasets/nih-chest-xrays/sample",
                data_dir=data_dir
            )
            # Reorganize
            sample_dir = os.path.join(data_dir, "sample")
            if os.path.exists(sample_dir):
                # Move files to nih directory
                for f in os.listdir(sample_dir):
                    src = os.path.join(sample_dir, f)
                    dst = os.path.join(output_dir, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                # Rename sample_images to images
                sample_images = os.path.join(output_dir, "sample", "images")
                if os.path.exists(sample_images) and not os.path.exists(images_dir):
                    shutil.move(sample_images, images_dir)
        else:
            print("   Downloading full NIH ChestX-ray14 dataset...")
            print("   WARNING: This is ~45 GB and will take a long time!")
            od.download(
                "https://www.kaggle.com/datasets/nih-chest-xrays/data",
                data_dir=data_dir
            )
            downloaded = os.path.join(data_dir, "data")
            if os.path.exists(downloaded):
                for f in os.listdir(downloaded):
                    src = os.path.join(downloaded, f)
                    dst = os.path.join(output_dir, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
        
        print("   NIH dataset download complete!")
        return True
        
    except Exception as e:
        print(f"   Download failed: {e}")
        return False


def download_shenzhen_tb(data_dir):
    """Download Shenzhen TB dataset."""
    print("=" * 60)
    print("2. SHENZHEN TB DATASET")
    print("=" * 60)
    print("   662 frontal CXRs (326 normal, 336 TB positive)")
    print("   Size: ~100 MB")
    print()
    
    output_dir = os.path.join(data_dir, "shenzhen")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    if os.path.exists(images_dir):
        img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg'))])
        if img_count > 100:
            print(f"   Already downloaded ({img_count} images). Skipping.")
            return True
    
    try:
        import opendatasets as od
        
        print("   Downloading from Kaggle...")
        od.download(
            "https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen",
            data_dir=data_dir
        )
        
        # Reorganize downloaded files
        dl_dir = os.path.join(data_dir, "tuberculosis-chest-xrays-shenzhen")
        if os.path.exists(dl_dir):
            for root, dirs, files in os.walk(dl_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src = os.path.join(root, f)
                        dst = os.path.join(images_dir, f)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
        
        # Create labels
        create_shenzhen_labels(output_dir, images_dir)
        
        img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg'))])
        print(f"   Shenzhen TB download complete! ({img_count} images)")
        return True
        
    except Exception as e:
        print(f"   Kaggle download failed: {e}")
        print("   Trying direct NLM download...")
        return download_shenzhen_direct(data_dir)


def download_shenzhen_direct(data_dir):
    """Try downloading Shenzhen TB from NLM directly."""
    import urllib.request
    
    output_dir = os.path.join(data_dir, "shenzhen")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    urls_to_try = [
        "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/ChinaSet_AllFiles.zip",
        "https://lhncbc.nlm.nih.gov/LHC-downloads/downloads/ChinaSet_AllFiles.zip",
    ]
    
    for url in urls_to_try:
        try:
            zip_path = os.path.join(output_dir, "download.zip")
            print(f"   Trying: {url}")
            urllib.request.urlretrieve(url, zip_path)
            
            # Check if it's actually a zip
            import zipfile
            if zipfile.is_zipfile(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(output_dir)
                os.remove(zip_path)
                
                # Find and move images
                for root, dirs, files in os.walk(output_dir):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg')) and root != images_dir:
                            src = os.path.join(root, f)
                            dst = os.path.join(images_dir, f)
                            if not os.path.exists(dst):
                                shutil.move(src, dst)
                
                create_shenzhen_labels(output_dir, images_dir)
                return True
            else:
                os.remove(zip_path)
                
        except Exception as e:
            print(f"   Failed: {e}")
            continue
    
    print("   All download methods failed for Shenzhen TB.")
    print("   Manual download: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html")
    return False


def create_shenzhen_labels(output_dir, images_dir):
    """Create labels CSV for Shenzhen TB dataset."""
    labels_path = os.path.join(output_dir, "labels.csv")
    
    records = []
    for img_name in sorted(os.listdir(images_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Shenzhen naming: CHNCXR_NNNN_0.png = normal, CHNCXR_NNNN_1.png = TB
        is_tb = 1 if '_1.' in img_name or '_1_' in img_name else 0
        records.append({
            'filename': img_name,
            'pneumonia': 0,
            'tb': is_tb,
            'normal': 1 if is_tb == 0 else 0
        })
    
    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'pneumonia', 'tb', 'normal'])
        writer.writeheader()
        writer.writerows(records)
    
    tb_count = sum(1 for r in records if r['tb'] == 1)
    normal_count = sum(1 for r in records if r['normal'] == 1)
    print(f"   Labels: {tb_count} TB, {normal_count} Normal -> {labels_path}")


def download_rsna_pneumonia(data_dir, small=False):
    """Download RSNA Pneumonia Detection dataset from Kaggle."""
    print("=" * 60)
    print("3. RSNA PNEUMONIA DETECTION DATASET")
    print("=" * 60)
    print("   26,684 chest X-rays with pneumonia bounding boxes")
    print("   Size: ~30 GB (full)")
    print()
    
    output_dir = os.path.join(data_dir, "rsna")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "stage_2_train_labels.csv")
    if os.path.exists(csv_path):
        print("   Already downloaded. Skipping.")
        return True
    
    try:
        import opendatasets as od
        
        if small:
            print("   Downloading RSNA dataset (competition files)...")
        else:
            print("   Downloading full RSNA Pneumonia dataset...")
            print("   WARNING: This is ~30 GB and will take a long time!")
        
        od.download(
            "https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge",
            data_dir=data_dir
        )
        
        # Reorganize
        dl_dir = os.path.join(data_dir, "rsna-pneumonia-detection-challenge")
        if os.path.exists(dl_dir):
            for f in os.listdir(dl_dir):
                src = os.path.join(dl_dir, f)
                dst = os.path.join(output_dir, f)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
        
        print("   RSNA download complete!")
        return True
        
    except Exception as e:
        print(f"   Download failed: {e}")
        print("   You may need to accept competition rules at:")
        print("   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules")
        return False


def print_summary(data_dir):
    """Print summary of downloaded datasets."""
    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    total = 0
    
    # NIH
    nih_dir = os.path.join(data_dir, "nih", "images")
    if os.path.exists(nih_dir):
        count = len([f for f in os.listdir(nih_dir) if f.endswith('.png')])
        print(f"  NIH ChestX-ray14:    {count:>6} images")
        total += count
    else:
        print(f"  NIH ChestX-ray14:    NOT FOUND")
    
    # Shenzhen
    shenzhen_dir = os.path.join(data_dir, "shenzhen", "images")
    if os.path.exists(shenzhen_dir):
        count = len([f for f in os.listdir(shenzhen_dir) if f.lower().endswith(('.png', '.jpg'))])
        print(f"  Shenzhen TB:         {count:>6} images")
        total += count
    else:
        print(f"  Shenzhen TB:         NOT FOUND")
    
    # RSNA
    rsna_dir = os.path.join(data_dir, "rsna")
    rsna_csv = os.path.join(rsna_dir, "stage_2_train_labels.csv")
    if os.path.exists(rsna_csv):
        import pandas as pd
        df = pd.read_csv(rsna_csv)
        count = df['patientId'].nunique()
        print(f"  RSNA Pneumonia:      {count:>6} images")
        total += count
    else:
        print(f"  RSNA Pneumonia:      NOT FOUND")
    
    print(f"  {'':>19}  ------")
    print(f"  Total:               {total:>6} images")
    
    print()
    print("NEXT STEPS:")
    print("  Train on the downloaded data:")
    print(f"  python train.py --data_dir {data_dir} --epochs 20 --batch_size 16 --num_workers 0")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Download chest X-ray datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['nih', 'shenzhen', 'rsna', 'all'],
                        help='Which dataset to download')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--small', action='store_true',
                        help='Download small sample instead of full dataset')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    print()
    print("Chest X-Ray Dataset Downloader")
    print("=" * 60)
    print(f"Data directory: {os.path.abspath(args.data_dir)}")
    print()
    
    # Check credentials
    if not check_kaggle_credentials():
        sys.exit(1)
    
    print()
    
    if args.dataset in ('shenzhen', 'all'):
        download_shenzhen_tb(args.data_dir)
        print()
    
    if args.dataset in ('nih', 'all'):
        download_nih_chestxray14(args.data_dir, small=args.small)
        print()
    
    if args.dataset in ('rsna', 'all'):
        download_rsna_pneumonia(args.data_dir, small=args.small)
        print()
    
    print_summary(args.data_dir)


if __name__ == '__main__':
    main()
