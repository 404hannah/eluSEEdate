import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import gc
import os
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Custom project imports
from models.conv_lstm_classifier import ConvLSTMModel
from dataset import MVOVideoDataset
from utils import *

class Tester:
    """
    It loads a trained model, feeds it unseen data, 
    and records how accurately and how fast the model makes decisions.
    """
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((HEIGHT, WIDTH))
        ])
        
        self.model = ConvLSTMModel(
            input_dim=6,
            hidden_dim=[64, 32],
            kernel_size=(3, 3),
            num_layers=2,
            height=HEIGHT,
            width=WIDTH,
            num_classes=3,
            dropout_rate=0.5  # Dropout not applied during eval mode
        ).to(self.device)
        
        # 3-channel model 
        # self.model = ConvLSTMModel(
        #     input_dim=3,
        #     hidden_dim=[64, 32],
        #     kernel_size=(3, 3),
        #     num_layers=2,
        #     height=HEIGHT,
        #     width=WIDTH,
        #     num_classes=3,
        #     dropout_rate=0.5
        # ).to(self.device)
        
        # Load the Weights
        self.load_weights()

    def load_weights(self):
        # Attempts to load the best model file and sets it to evaluation mode
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Did you run train.py?")

    def test(self):
        # The main evaluation loop. 
        # Processes data one-by-one to measure real-world performance
        print("Preparing Test Data...")

        # Splitfolders approach
        test_dir = os.path.join("output", "test")
        test_dir_vid = os.path.join(test_dir, "videos")
        test_lbl_vid = os.path.join(test_dir, "labels")
        test_dataset = MVOVideoDataset(test_dir_vid, test_lbl_vid, transforms=self.transforms)
        
        test_dataset.set_split_type('TEST', len(test_dataset))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
        
        print(f"Evaluating {len(test_dataset)} videos and measuring latency...")
        
        # full_dataset = MVOVideoDataset(VIDEO_DIR, LABEL_DIR, transforms=self.transforms)
        # 
        # SEED = 8
        # total_size = len(full_dataset)
        # train_size = int(0.6 * total_size)
        # val_size = int(0.2 * total_size)
        # test_size = total_size - train_size - val_size
        #
        # generator = torch.Generator().manual_seed(SEED)
        # 
        # _, _, test_dataset = random_split(
        #     full_dataset, 
        #     [train_size, val_size, test_size], 
        #     generator=generator
        # )
        # 
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
        # print(f"Evaluating {len(test_dataset)} videos and measuring latency...")
        
        all_preds = []
        all_labels = []
        latencies = []
        
        # Evaluation Loop
        with torch.no_grad(): 
            for i, (video_tensor, labels) in enumerate(tqdm(test_loader, leave=True)):
                video_tensor = video_tensor.float().to(self.device)
                labels = labels.to(self.device)
                
                # Latency Measurement Start
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    # Wait for the gpu to be ready 
                    # Also lets just assume may cuda / gpu tayo rn
                    # We can just fix later and stuff
                
                start_time = time.perf_counter() # Timer
                
                outputs = self.model(video_tensor) # Forward pass (The Inference)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize() # Wait for the GPU to finish the math
                
                end_time = time.perf_counter()
                # Latency Measurement End
                
                # We skip the first 5 frames ('warm-up') because the system 
                # is often slow during its first few calculations.
                if i >= 5:
                    latencies.append(end_time - start_time)

                # Convert raw scores to the predicted class index (0, 1, or 2)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Memory cleanup: Delete intermediate tensors
                del video_tensor, labels, outputs, predicted
                
                # Periodic cache clearing every 50 videos
                if i % 50 == 0:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

        # Calculate Latency Stats
        avg_latency_ms = np.mean(latencies) * 1000
        inf_fps = 1 / np.mean(latencies) if len(latencies) > 0 else 0
        
        # Calculate and Print all results
        self.calculate_metrics(all_labels, all_preds, avg_latency_ms, inf_fps)
        
        # Save detailed logs to a CSV
        self.save_results(all_labels, all_preds)

    def calculate_metrics(self, y_true, y_pred, avg_latency_ms, inf_fps):
        # Computes statistical performance and prints the Final Report
        print(f"Avg Latency:        {avg_latency_ms:.2f} ms per video clip")
        print(f"Inference Speed:    {inf_fps:.2f} clips per second")
        # Computes statistical performance and prints the Final Report
        print("\n" + "-"*40)
        print("       FINAL PERFORMANCE REPORT       ")
        print("-"*40)
        
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy:   {acc*100:.2f}%")
        
        # Precision and Recall
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"Precision:          {precision:.4f}")
        print(f"Recall:             {recall:.4f}")
        print("-" * 40)
       
        print(f"Avg Latency:        {avg_latency_ms:.2f} ms per video clip")
        print(f"Inference Speed:    {inf_fps:.2f} clips per second")
       
        print("-" * 40)
        print("Detailed Class Report:")
        # Generates a table for Front(0), Left(1), Right(2)
        print(classification_report(y_true, y_pred, target_names=['Front', 'Left', 'Right'], zero_division=0))

    def save_results(self, y_true, y_pred):
        # Creates a CSV to see exactly which videos failed
        df = pd.DataFrame({
            'Actual_Label': y_true,
            'Predicted_Label': y_pred
        })
        
        # Map numbers back to words for readability
        label_map = {0: 'Front', 1: 'Left', 2: 'Right'}
        df['Actual_Text'] = df['Actual_Label'].map(label_map)
        df['Predicted_Text'] = df['Predicted_Label'].map(label_map)
        
        # Check if correct
        df['Correct'] = df['Actual_Label'] == df['Predicted_Label']
        
        save_path = "test_results.csv"
        df.to_csv(save_path, index=False)
        print(f"\nDetailed predictions saved to '{save_path}'")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "best_convlstm.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run the Tester
    tester = Tester(MODEL_PATH, DEVICE)
    tester.test()