"""
Dataset Download & Preparation Script
Downloads real chest X-ray datasets for training.

Supported datasets:
1. Shenzhen TB Dataset (small, ~100MB) - Downloads automatically
2. NIH ChestX-ray14 (large, ~45GB) - Provides instructions
3. RSNA Pneumonia (large, ~30GB) - Provides Kaggle instructions

Usage:
    python download_datasets.py --dataset shenzhen
    python download_datasets.py --dataset all
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path


def download_file(url, dest_path, description="file"):
    """Download a file with progress bar."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest_path, progress_hook)
    print("\n  Download complete!")


def setup_shenzhen_tb(data_dir):
    """
    Download and prepare the Shenzhen TB dataset.
    Source: National Library of Medicine
    ~662 frontal chest X-rays (326 normal, 336 TB)
    """
    dataset_dir = os.path.join(data_dir, "shenzhen")
    images_dir = os.path.join(dataset_dir, "images")
    
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 100:
        print("Shenzhen TB dataset already exists. Skipping download.")
        return True
    
    print("=" * 60)
    print("SHENZHEN TB DATASET")
    print("=" * 60)
    print("Source: U.S. National Library of Medicine")
    print("Size: ~100 MB")
    print("Images: 662 frontal CXRs (326 normal, 336 TB positive)")
    print()
    
    os.makedirs(images_dir, exist_ok=True)
    
    # The Shenzhen dataset is available from the NLM
    # Direct download URL
    zip_url = "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip"
    zip_path = os.path.join(dataset_dir, "ChinaSet_AllFiles.zip")
    
    try:
        download_file(zip_url, zip_path, "Shenzhen TB Dataset")
        
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dataset_dir)
        
        # Move images to the expected location
        extracted_dir = os.path.join(dataset_dir, "ChinaSet_AllFiles", "CXR_png")
        if os.path.exists(extracted_dir):
            for f in os.listdir(extracted_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.move(os.path.join(extracted_dir, f), os.path.join(images_dir, f))
        
        # Clean up zip
        os.remove(zip_path)
        
        # Count images
        count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg'))])
        print(f"  Successfully prepared {count} images in {images_dir}")
        
        # Create labels file for easier loading
        create_shenzhen_labels(dataset_dir, images_dir)
        
        return True
        
    except Exception as e:
        print(f"\n  Auto-download failed: {e}")
        print("\n  MANUAL DOWNLOAD INSTRUCTIONS:")
        print("  1. Visit: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#702702dc-9e4e-4e00-ae82-1a4e3bdd0b28")
        print("  2. Download 'ChinaSet_AllFiles.zip'")
        print(f"  3. Extract PNG images to: {images_dir}")
        print("  4. Re-run this script")
        return False


def create_shenzhen_labels(dataset_dir, images_dir):
    """Create a CSV labels file for the Shenzhen dataset."""
    import csv
    
    labels_path = os.path.join(dataset_dir, "labels.csv")
    
    with open(labels_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'tb', 'normal'])
        
        for img_name in sorted(os.listdir(images_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            # Shenzhen naming convention:
            # CHNCXR_NNNN_0.png = normal
            # CHNCXR_NNNN_1.png = TB positive
            if '_1.' in img_name or '_1_' in img_name:
                writer.writerow([img_name, 1, 0])
            else:
                writer.writerow([img_name, 0, 1])
    
    print(f"  Labels saved to {labels_path}")


def setup_nih_instructions(data_dir):
    """Print instructions for downloading NIH ChestX-ray14."""
    dataset_dir = os.path.join(data_dir, "nih")
    
    print("=" * 60)
    print("NIH CHESTX-RAY14 DATASET")
    print("=" * 60)
    print("Size: ~45 GB (112,120 frontal chest X-rays)")
    print("Labels: 14 pathology classes including Pneumonia")
    print()
    print("DOWNLOAD INSTRUCTIONS:")
    print()
    print("  Option A - NIH Box (Recommended):")
    print("  1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC")
    print("  2. Download all image zip files (images_001.tar.gz to images_012.tar.gz)")
    print("  3. Download 'Data_Entry_2017.csv'")
    print(f"  4. Extract all images to: {os.path.join(dataset_dir, 'images')}")
    print(f"  5. Place CSV at: {os.path.join(dataset_dir, 'Data_Entry_2017.csv')}")
    print()
    print("  Option B - Kaggle:")
    print("  1. Visit: https://www.kaggle.com/datasets/nih-chest-xrays/data")
    print("  2. Download and extract")
    print(f"  3. Organize into: {dataset_dir}")
    print()
    print(f"  Expected structure:")
    print(f"    {dataset_dir}/")
    print(f"      Data_Entry_2017.csv")
    print(f"      images/")
    print(f"        00000001_000.png")
    print(f"        00000001_001.png")
    print(f"        ...")
    print()
    
    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)


def setup_rsna_instructions(data_dir):
    """Print instructions for downloading RSNA Pneumonia dataset."""
    dataset_dir = os.path.join(data_dir, "rsna")
    
    print("=" * 60)
    print("RSNA PNEUMONIA DETECTION DATASET")
    print("=" * 60)
    print("Size: ~30 GB (26,684 chest X-rays with bounding boxes)")
    print("Source: Kaggle Competition")
    print()
    print("DOWNLOAD INSTRUCTIONS:")
    print()
    print("  1. Install Kaggle CLI: pip install kaggle")
    print("  2. Set up Kaggle API credentials (~/.kaggle/kaggle.json)")
    print("  3. Run:")
    print("     kaggle competitions download -c rsna-pneumonia-detection-challenge")
    print(f"  4. Extract to: {dataset_dir}")
    print()
    print("  OR manually:")
    print("  1. Visit: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data")
    print("  2. Download 'stage_2_train_labels.csv' and 'stage_2_train_images'")
    print(f"  3. Place in: {dataset_dir}")
    print()
    print(f"  Expected structure:")
    print(f"    {dataset_dir}/")
    print(f"      stage_2_train_labels.csv")
    print(f"      stage_2_train_images/")
    print(f"        patient_id.dcm")
    print(f"        ...")
    print()
    
    os.makedirs(os.path.join(dataset_dir, "stage_2_train_images"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Download chest X-ray datasets')
    parser.add_argument('--dataset', type=str, default='shenzhen',
                        choices=['shenzhen', 'nih', 'rsna', 'all'],
                        help='Which dataset to download/setup')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root data directory')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    print()
    print("Chest X-Ray Dataset Setup")
    print("=" * 60)
    print(f"Data directory: {os.path.abspath(args.data_dir)}")
    print()
    
    if args.dataset in ('shenzhen', 'all'):
        setup_shenzhen_tb(args.data_dir)
        print()
    
    if args.dataset in ('nih', 'all'):
        setup_nih_instructions(args.data_dir)
        print()
    
    if args.dataset in ('rsna', 'all'):
        setup_rsna_instructions(args.data_dir)
        print()
    
    print("=" * 60)
    print("NEXT STEPS:")
    print("  After downloading datasets, train the model:")
    print(f"  python train.py --data_dir {args.data_dir} --epochs 30 --batch_size 32")
    print("=" * 60)


if __name__ == '__main__':
    main()
