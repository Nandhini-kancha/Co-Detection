"""
Model Performance Evaluation Script
Generates comprehensive metrics, confusion matrix, and plots.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


def load_training_history(path):
    with open(path, 'r') as f:
        return json.load(f)


def plot_training_curves(history, save_dir):
    """Plot loss and AUC curves over epochs."""
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_auc = [h['train_auc'] for h in history]
    val_auc = [h['val_auc'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Training Performance', fontsize=14, fontweight='bold')
    
    # Loss curve
    axes[0].plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, val_loss, 'r-o', label='Val Loss', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC curve
    axes[1].plot(epochs, train_auc, 'b-o', label='Train AUC', markersize=4)
    axes[1].plot(epochs, val_auc, 'r-o', label='Val AUC', markersize=4)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('Training & Validation AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.3, 1.0)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_per_class_auc(history, save_dir):
    """Plot per-class AUC progression."""
    epochs = [h['epoch'] for h in history]
    classes = list(history[0]['per_class_auc'].keys())
    colors = ['#EF4444', '#3B82F6', '#10B981']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for cls, color in zip(classes, colors):
        aucs = [h['per_class_auc'][cls] for h in history]
        ax.plot(epochs, aucs, '-o', label=cls, color=color, markersize=4)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Per-Class AUC Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.1, 1.0)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'per_class_auc.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_final_performance_bar(history, save_dir):
    """Bar chart of best per-class AUC."""
    # Find best epoch
    best_epoch = max(history, key=lambda h: h['val_auc'])
    
    classes = list(best_epoch['per_class_auc'].keys())
    aucs = [best_epoch['per_class_auc'][cls] for cls in classes]
    colors = ['#EF4444', '#3B82F6', '#10B981']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, aucs, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{auc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_ylabel('AUC-ROC')
    ax.set_title(f'Best Model Performance (Epoch {best_epoch["epoch"]}, Val AUC: {best_epoch["val_auc"]:.4f})')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'final_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def generate_report(history):
    """Print a text summary of training performance."""
    best = max(history, key=lambda h: h['val_auc'])
    last = history[-1]
    first = history[0]
    
    print()
    print("=" * 60)
    print("MODEL PERFORMANCE REPORT")
    print("=" * 60)
    
    print(f"\n  Training: {len(history)} epochs")
    print(f"  Training data: 80 samples (synthetic random noise)")
    print(f"  Validation data: 20 samples (synthetic random noise)")
    
    print(f"\n--- Best Epoch: {best['epoch']} ---")
    print(f"  Val Loss:  {best['val_loss']:.4f}")
    print(f"  Val AUC:   {best['val_auc']:.4f}")
    print(f"  Per-class AUC:")
    for cls, auc in best['per_class_auc'].items():
        status = "GOOD" if auc > 0.7 else "POOR" if auc > 0.55 else "RANDOM"
        print(f"    {cls:15s}: {auc:.4f}  [{status}]")
    
    print(f"\n--- Final Epoch: {last['epoch']} ---")
    print(f"  Train Loss: {last['train_loss']:.4f}")
    print(f"  Train AUC:  {last['train_auc']:.4f}")
    print(f"  Val Loss:   {last['val_loss']:.4f}")
    print(f"  Val AUC:    {last['val_auc']:.4f}")
    
    print(f"\n--- Improvement over training ---")
    print(f"  Val AUC: {first['val_auc']:.4f} -> {best['val_auc']:.4f} ({best['val_auc'] - first['val_auc']:+.4f})")
    print(f"  Train Loss: {first['train_loss']:.4f} -> {last['train_loss']:.4f} ({last['train_loss'] - first['train_loss']:+.4f})")
    
    # Diagnosis
    print(f"\n--- DIAGNOSIS ---")
    if best['val_auc'] < 0.65:
        print("  [!!] Model performance is POOR (AUC < 0.65)")
        print("  [!!] The model was trained on SYNTHETIC RANDOM DATA (100 images)")
        print("  [!!] Random noise images cannot teach meaningful features")
        print()
        print("  ROOT CAUSE: No real chest X-ray data was used for training.")
        print("  SOLUTION:   Download real datasets and retrain:")
        print("    1. python download_datasets.py --dataset shenzhen")
        print("    2. python train.py --data_dir ./data --epochs 30 --batch_size 16")
    elif best['val_auc'] < 0.80:
        print("  [!] Model performance is MODERATE (AUC 0.65-0.80)")
        print("  Consider: more training data, more epochs, or data augmentation")
    else:
        print("  [OK] Model performance is GOOD (AUC > 0.80)")
    
    print("=" * 60)


def main():
    history_path = os.path.join('backend', 'checkpoints', 'training_history.json')
    
    if not os.path.exists(history_path):
        print("No training history found. Train the model first:")
        print("  python train.py --epochs 20 --batch_size 16 --num_workers 0")
        return
    
    history = load_training_history(history_path)
    
    # Create output directory
    plots_dir = os.path.join('backend', 'checkpoints', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Generating performance plots...")
    plot_training_curves(history, plots_dir)
    plot_per_class_auc(history, plots_dir)
    plot_final_performance_bar(history, plots_dir)
    
    generate_report(history)


if __name__ == '__main__':
    main()
