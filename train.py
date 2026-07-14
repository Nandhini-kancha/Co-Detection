"""
Training script for multi-label Chest X-ray classifier.
Supports: NIH ChestX-ray14, Shenzhen TB, RSNA Pneumonia datasets.

Usage:
    python train.py --data_dir ./data --epochs 30 --batch_size 32 --lr 1e-4
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import time

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from model import ChestXrayModel, LABEL_NAMES


# ─── Dataset Classes ───────────────────────────────────────────────────

class UnifiedChestXrayDataset(Dataset):
    """Unified dataset that merges NIH, Shenzhen TB, and RSNA data."""
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a black image if file is corrupt
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor([
            float(row['pneumonia']),
            float(row['tb']),
            float(row['normal'])
        ], dtype=torch.float32)
        
        return image, labels


# ─── Data Loading Functions ────────────────────────────────────────────

def load_nih_chestxray14(data_dir):
    """Load NIH ChestX-ray14 dataset."""
    csv_path = os.path.join(data_dir, 'nih', 'Data_Entry_2017.csv')
    image_dir = os.path.join(data_dir, 'nih', 'images')
    
    if not os.path.exists(csv_path):
        print(f"NIH dataset not found at {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    records = []
    
    for _, row in df.iterrows():
        labels = str(row['Finding Labels']).split('|')
        has_pneumonia = 1 if 'Pneumonia' in labels else 0
        is_normal = 1 if 'No Finding' in labels else 0
        
        records.append({
            'image_path': os.path.join(image_dir, row['Image Index']),
            'pneumonia': has_pneumonia,
            'tb': 0,  # NIH doesn't have TB labels
            'normal': is_normal,
            'source': 'nih'
        })
    
    return pd.DataFrame(records)


def load_shenzhen_tb(data_dir):
    """Load Shenzhen TB dataset."""
    tb_dir = os.path.join(data_dir, 'shenzhen')
    images_dir = os.path.join(tb_dir, 'images')
    
    if not os.path.exists(images_dir):
        print(f"Shenzhen TB dataset not found at {tb_dir}")
        return pd.DataFrame()
    
    records = []
    
    # Try loading from labels CSV first (more reliable)
    labels_csv = os.path.join(tb_dir, 'labels.csv')
    if os.path.exists(labels_csv):
        labels_df = pd.read_csv(labels_csv)
        for _, row in labels_df.iterrows():
            img_path = os.path.join(images_dir, row['filename'])
            if os.path.exists(img_path):
                records.append({
                    'image_path': img_path,
                    'pneumonia': 0,
                    'tb': int(row['tb']),
                    'normal': int(row['normal']),
                    'source': 'shenzhen'
                })
    else:
        # Fallback: use filename convention
        # CHNCXR_NNNN_0.png = normal, CHNCXR_NNNN_1.png = TB positive
        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            is_tb = 1 if '_1.' in img_name or '_1_' in img_name else 0
            
            records.append({
                'image_path': os.path.join(images_dir, img_name),
                'pneumonia': 0,
                'tb': is_tb,
                'normal': 1 if is_tb == 0 else 0,
                'source': 'shenzhen'
            })
    
    return pd.DataFrame(records)


def load_rsna_pneumonia(data_dir):
    """Load RSNA Pneumonia Detection dataset."""
    csv_path = os.path.join(data_dir, 'rsna', 'stage_2_train_labels.csv')
    image_dir = os.path.join(data_dir, 'rsna', 'stage_2_train_images')
    
    if not os.path.exists(csv_path):
        print(f"RSNA dataset not found at {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    # RSNA has duplicate rows for bounding boxes, take unique patient IDs
    df_unique = df.groupby('patientId')['Target'].max().reset_index()
    
    records = []
    for _, row in df_unique.iterrows():
        img_path = os.path.join(image_dir, f"{row['patientId']}.dcm")
        has_pneumonia = int(row['Target'])
        
        records.append({
            'image_path': img_path,
            'pneumonia': has_pneumonia,
            'tb': 0,
            'normal': 1 if has_pneumonia == 0 else 0,
            'source': 'rsna'
        })
    
    return pd.DataFrame(records)


def create_unified_dataset(data_dir):
    """Merge all datasets into a unified format."""
    print("Loading datasets...")
    
    # Check for pre-built unified CSV (from download_real_data.py)
    unified_csv = os.path.join(data_dir, 'unified_labels.csv')
    if os.path.exists(unified_csv):
        print(f"  Found unified labels at {unified_csv}")
        df = pd.read_csv(unified_csv)
        # Filter to only existing images
        df = df[df['image_path'].apply(os.path.exists)]
        if len(df) > 0:
            print(f"  Unified dataset: {len(df)} images")
            print(f"    Pneumonia: {df['pneumonia'].astype(int).sum()}")
            print(f"    TB: {df['tb'].astype(int).sum()}")
            print(f"    Normal: {df['normal'].astype(int).sum()}")
            return df
    
    dfs = []
    
    # Load HuggingFace-downloaded pneumonia dataset
    pneumonia_labels = os.path.join(data_dir, 'pneumonia', 'labels.csv')
    if os.path.exists(pneumonia_labels):
        pn_df = pd.read_csv(pneumonia_labels)
        pn_df['image_path'] = pn_df['filename'].apply(
            lambda f: os.path.join(data_dir, 'pneumonia', 'images', f))
        pn_df['source'] = 'pneumonia_hf'
        pn_df = pn_df[pn_df['image_path'].apply(os.path.exists)]
        if len(pn_df) > 0:
            print(f"  Pneumonia (HuggingFace): {len(pn_df)} images")
            dfs.append(pn_df[['image_path', 'pneumonia', 'tb', 'normal', 'source']])
    
    # Load HuggingFace-downloaded TB dataset
    tb_labels = os.path.join(data_dir, 'tb_hf', 'labels.csv')
    if os.path.exists(tb_labels):
        tb_df = pd.read_csv(tb_labels)
        tb_df['image_path'] = tb_df['filename'].apply(
            lambda f: os.path.join(data_dir, 'tb_hf', 'images', f))
        tb_df['source'] = 'tb_hf'
        tb_df = tb_df[tb_df['image_path'].apply(os.path.exists)]
        if len(tb_df) > 0:
            print(f"  TB (HuggingFace): {len(tb_df)} images")
            dfs.append(tb_df[['image_path', 'pneumonia', 'tb', 'normal', 'source']])
    
    # Load original datasets
    nih_df = load_nih_chestxray14(data_dir)
    if len(nih_df) > 0:
        print(f"  NIH ChestX-ray14: {len(nih_df)} images")
        dfs.append(nih_df)
    
    shenzhen_df = load_shenzhen_tb(data_dir)
    if len(shenzhen_df) > 0:
        print(f"  Shenzhen TB: {len(shenzhen_df)} images")
        dfs.append(shenzhen_df)
    
    rsna_df = load_rsna_pneumonia(data_dir)
    if len(rsna_df) > 0:
        print(f"  RSNA Pneumonia: {len(rsna_df)} images")
        dfs.append(rsna_df)
    
    if len(dfs) == 0:
        print("\nNo datasets found! Creating synthetic demo dataset...")
        return create_demo_dataset()
    
    unified = pd.concat(dfs, ignore_index=True)
    print(f"\nUnified dataset: {len(unified)} total images")
    print(f"  Pneumonia positive: {unified['pneumonia'].astype(int).sum()}")
    print(f"  TB positive: {unified['tb'].astype(int).sum()}")
    print(f"  Normal: {unified['normal'].astype(int).sum()}")
    
    return unified


def create_demo_dataset():
    """Create a small synthetic dataset for demo/testing."""
    print("Generating synthetic demo dataset (100 samples)...")
    demo_dir = os.path.join('data', 'demo')
    os.makedirs(demo_dir, exist_ok=True)
    
    records = []
    for i in range(100):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(demo_dir, f'demo_{i:04d}.png')
        img.save(img_path)
        
        # Random labels
        pneumonia = 1 if np.random.random() > 0.7 else 0
        tb = 1 if np.random.random() > 0.8 else 0
        normal = 1 if (pneumonia == 0 and tb == 0) else 0
        
        records.append({
            'image_path': img_path,
            'pneumonia': pneumonia,
            'tb': tb,
            'normal': normal,
            'source': 'demo'
        })
    
    return pd.DataFrame(records)


# ─── Training Functions ───────────────────────────────────────────────

def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    avg_loss = running_loss / len(dataloader)
    
    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        auc = 0.0
    
    return avg_loss, auc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    avg_loss = running_loss / len(dataloader)
    
    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        auc = 0.0
    
    # Per-class AUC
    per_class_auc = {}
    for i, name in enumerate(LABEL_NAMES):
        try:
            per_class_auc[name] = round(roc_auc_score(all_labels[:, i], all_preds[:, i]), 4)
        except ValueError:
            per_class_auc[name] = 0.0
    
    return avg_loss, auc, per_class_auc


def main():
    parser = argparse.ArgumentParser(description='Train Chest X-ray Multi-label Classifier')
    parser.add_argument('--data_dir', type=str, default='./data', help='Root data directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--checkpoint_dir', type=str, default='./backend/checkpoints')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    df = create_unified_dataset(args.data_dir)
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['source'] if 'source' in df.columns else None)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}")
    
    # Datasets & Loaders
    train_dataset = UnifiedChestXrayDataset(train_df, transform=get_train_transforms(args.image_size))
    val_dataset = UnifiedChestXrayDataset(val_df, transform=get_val_transforms(args.image_size))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Model
    model = ChestXrayModel(num_classes=3, pretrained=True).to(device)
    
    # Loss with class weights for imbalance
    pos_counts = train_df[['pneumonia', 'tb', 'normal']].sum().values
    neg_counts = len(train_df) - pos_counts
    pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-5), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_auc = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, per_class_auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
        print(f"Per-class AUC: {per_class_auc}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'per_class_auc': per_class_auc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"  [*] Best model saved (AUC: {best_auc:.4f})")
        
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'latest_model.pth'))
    
    # Save training history
    with open(os.path.join(args.checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best validation AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
