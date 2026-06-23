import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import random 
import splitfolders
import os
import time
import gc

# Custom project imports
from models.conv_lstm_classifier import ConvLSTMModel
from dataset import MVOVideoDataset
from utils import *

# Configurations
BATCH = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
SAVED_MODEL_PATH = "best_convlstm.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early Stopping Configuration
EARLY_STOP_PATIENCE = 5  # Stop if no improvement for 5 epochs
MIN_DELTA = 0.01  # Minimum change to qualify as improvement (0.01%)

# Setting a fixed random seed to ensure that 
# we get the exact same data split every time we run the script
SEED = 8
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Model Parameters
PARAMS = {
    'input_dim': 6,
    'hidden_dim': [64, 32], 
    'kernel_size': (3, 3),
    'num_layers': 2,
    'height': HEIGHT,
    'width': WIDTH,
    'num_classes': 3 
}

# Proto 2: 3-channel RGB-only (Eric's version - COMMENTED OUT)
# PARAMS = {
#     'input_dim': 3,
#     'hidden_dim': [64, 32], 
#     'kernel_size': (3, 3),
#     'num_layers': 2,
#     'height': HEIGHT,
#     'width': WIDTH,
#     'num_classes': 3 
# }

def train_one_epoch(model, loader, criterion, optimizer, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm = 0.0
    
    loop = tqdm(loader, leave=True)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping: Prevent exploding gradients in ConvLSTM
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        total_grad_norm += grad_norm.item()
        
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, predictions = scores.max(1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        
        loop.set_description(f"Loss: {loss.item():.4f} | GradNorm: {grad_norm.item():.3f}")
        
        # Memory cleanup: Delete intermediate tensors
        del data, targets, scores, loss, predictions
    
    # Clear GPU cache after epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    avg_grad_norm = total_grad_norm / len(loader)
    return running_loss / len(loader), 100 * correct / total, avg_grad_norm

def main():
    # Data Setup
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((HEIGHT, WIDTH))
    ])

    # Split the files in the directory into three directories: training (60%), validation (20%), and testing (20%)
    splitfolders.ratio(DATA_DIR, output="output", seed=8, ratio=(.6, .2, .2), group="sibling", move=False, shuffle=True) 

    train_dir = os.path.join("output", "train")
    val_dir = os.path.join("output", "val")
    test_dir = os.path.join("output", "test")

    train_dir_vid = os.path.join(train_dir, "videos")
    val_dir_vid = os.path.join(val_dir, "videos")
    test_dir_vid = os.path.join(test_dir, "videos")

    train_lbl_vid = os.path.join(train_dir, "labels")
    val_lbl_vid = os.path.join(val_dir, "labels")
    test_lbl_vid = os.path.join(test_dir, "labels")

    train_dataset = MVOVideoDataset(train_dir_vid, train_lbl_vid, transforms=transforms_train)
    val_dataset = MVOVideoDataset(val_dir_vid, val_lbl_vid, transforms=transforms_train)
    test_dataset = MVOVideoDataset(test_dir_vid, test_lbl_vid, transforms=transforms_train)

    print(f"Data Split -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test (Unused): {len(test_dataset)}")
    
    train_dataset.set_split_type('TRAIN', len(train_dataset))
    val_dataset.set_split_type('VALIDATION', len(val_dataset))

    # Calculate class instances for class weights
    label_counts, total_count = train_dataset.class_counter()
    
    # Add 1 to avoid division by zero
    front_weight = total_count / (label_counts[0] + 1) 
    left_weight = total_count / (label_counts[1] + 1)
    right_weight = total_count / (label_counts[2] + 1)

    print(f"Front class instances: {label_counts[0]} -> Front weight: {front_weight}")
    print(f"Left class instances: {label_counts[1]} -> Left weight: {left_weight}")
    print(f"Right class instances: {label_counts[2]} -> Right weight: {right_weight}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    # Model Setup
    model = ConvLSTMModel(
        input_dim=PARAMS['input_dim'],
        hidden_dim=PARAMS['hidden_dim'],
        kernel_size=PARAMS['kernel_size'],
        num_layers=PARAMS['num_layers'],
        height=PARAMS['height'],
        width=PARAMS['width'],
        num_classes=PARAMS['num_classes']
    ).to(DEVICE)

    # CrossEntropyLoss is used to handle class imbalance
    class_weights = torch.FloatTensor([front_weight, left_weight, right_weight]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # In-memory random splitting
    # transforms_train = transforms.Compose([
    #     transforms.Resize((HEIGHT, WIDTH)),
    #     transforms.ToTensor()
    # ])
    # 
    # full_dataset = MVOVideoDataset(VIDEO_DIR, LABEL_DIR, transforms=transforms_train)
    # 
    # total_size = len(full_dataset)
    # train_size = int(0.6 * total_size)
    # val_size = int(0.2 * total_size)
    # test_size = total_size - train_size - val_size
    # 
    # generator = torch.Generator().manual_seed(SEED)
    # 
    # train_dataset, val_dataset, _ = random_split(
    #     full_dataset, 
    #     [train_size, val_size, test_size], 
    #     generator=generator
    # )
    # 
    # print(f"Data Split -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test (Unused): {test_size}")
    # 
    # train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0)
    # 
    # model = ConvLSTMModel(
    #     input_dim=PARAMS['input_dim'],
    #     hidden_dim=PARAMS['hidden_dim'],
    #     kernel_size=PARAMS['kernel_size'],
    #     num_layers=PARAMS['num_layers'],
    #     height=PARAMS['height'],
    #     width=PARAMS['width'],
    #     num_classes=PARAMS['num_classes']
    # ).to(DEVICE)
    # 
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler: Reduces LR when validation accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # Maximize validation accuracy
        factor=0.5,           # Reduce LR by half
        patience=3,           # Wait 3 epochs before reducing
        min_lr=1e-7           # Don't go below this LR
    )

    # Training Loop with Early Stopping
    best_acc = 0
    epochs_no_improve = 0
    early_stop = False
    print(f"Training on {DEVICE} with {len(train_dataset)} videos.")
    print(f"Early stopping enabled: patience={EARLY_STOP_PATIENCE}, min_delta={MIN_DELTA}%")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_acc, avg_grad_norm = train_one_epoch(model, train_loader, criterion, optimizer, max_grad_norm=1.0)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_latencies = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = x.float().to(DEVICE)
                y = y.to(DEVICE)
                
                # Measure inference time
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                scores = model(x)
                
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                # Skip first batch for warm-up
                if batch_idx >= 1:
                    val_latencies.append(end_time - start_time)
                
                _, preds = scores.max(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                
                # Memory cleanup: Delete intermediate tensors
                del x, y, scores, preds
        
        # Clear GPU cache after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        val_acc = 100 * val_correct / val_total
        avg_val_latency_ms = (np.mean(val_latencies) * 1000) if len(val_latencies) > 0 else 0
        val_throughput = (1 / np.mean(val_latencies)) if len(val_latencies) > 0 else 0
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        print(f"Avg Gradient Norm: {avg_grad_norm:.4f} (clipped at 1.0)")
        print(f"Val Inference: {avg_val_latency_ms:.2f} ms/batch | {val_throughput:.2f} batches/sec")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)

        # Save Best Model & Early Stopping Check
        if val_acc > best_acc + MIN_DELTA:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVED_MODEL_PATH)
            print(f"✓ New best model saved! ({val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s). Best: {best_acc:.2f}%")
            
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n⚠ Early stopping triggered! No improvement for {EARLY_STOP_PATIENCE} epochs.")
                print(f"Best validation accuracy: {best_acc:.2f}%")
                early_stop = True
                break
    
    if not early_stop:
        print(f"\n✓ Training completed all {NUM_EPOCHS} epochs.")
    print(f"Final best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()