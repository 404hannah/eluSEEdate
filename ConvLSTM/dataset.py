import torch
import cv2
import pandas as pd
import os
from torch.utils.data import Dataset
import random
import numpy as np
import scipy.stats as ss
from PIL import Image

from utils import *

class MVOVideoDataset(Dataset):
    """
    This takes a 3-second video and turns it into
    a 'data packet' for the AI to study.
    """
    def __init__(self, video_folder, label_folder, transforms=None):
        self.video_folder = video_folder
        self.label_folder = label_folder
        self.transforms = transforms

        # Simple file listing
        self.video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        self.csv_files = [f.replace('.mp4', '.csv') for f in self.video_files]
        
        # Proto 2: Improved validation with multiple CSV naming conventions (Eric's version - COMMENTED OUT)
        # valid_video_files = []
        # valid_csv_files = []
        # try:
        #     all_video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        #     for video_name in all_video_files:
        #         video_path = os.path.join(video_folder, video_name)
        #
        #         # Try multiple CSV naming conventions
        #         csv_name = video_name.replace('.mp4', '.csv')
        #         csv_path = os.path.join(label_folder, csv_name)
        #         
        #         if not os.path.exists(csv_path):
        #             csv_name = video_name.replace('.mp4', '_labels.csv')
        #             csv_path = os.path.join(label_folder, csv_name)
        #         
        #         if not os.path.exists(csv_path):
        #             csv_name = video_name.replace('.mp4', '_labels_A.csv')
        #             csv_path = os.path.join(label_folder, csv_name)
        #
        #         if os.path.exists(video_path) and os.path.exists(csv_path):
        #             valid_video_files.append(video_name)
        #             valid_csv_files.append(csv_name)
        #         else:
        #             if not os.path.exists(video_path):
        #                 print(f"WARNING: Video file not found: {video_path}. Skipping.")
        #             if not os.path.exists(csv_path):
        #                 print(f"WARNING: Label file not found for video {video_name}. Skipping.")
        #
        #     self.video_files = valid_video_files
        #     self.csv_files = valid_csv_files
        #     print(f"Initialized dataset with {len(self.video_files)} valid video-label pairs.")
        #
        # except FileNotFoundError:
        #     print(f"Error: Could not find folder {video_folder} or {label_folder}.")
        #     self.video_files = []
        #     self.csv_files = []
        
        self.split_type = ''
        self.positions = []  # If split_type == 'train', this would not be filled

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_folder, video_name)
        
        # Get the Label from the matching CSV file
        csv_name = video_name.replace('.mp4', '.csv')
        csv_path = os.path.join(self.label_folder, csv_name)
        
        # Read the label
        # For a 3s clip, the label is the maximum turn label present in the last second (24 frames in CSV)
        df = pd.read_csv(csv_path)
        label = self.labeler(df)

        # Intent system
        if self.split_type == 'TRAIN':
            intent_position = self.get_intent_position()
        else:
            intent_position = self.positions[idx]

        intent = self.get_intent(intent_position, df)

        # Load the 30 frames from the 3-second video
        cap = cv2.VideoCapture(video_path)
        frames = []
        for i in range(30): # We already know it's exactly 30 frames
            ret, frame = cap.read()
            if not ret:
                # If a video is shorter than 3s, pad with a black frame
                frame = torch.zeros((3, 128, 128))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transforms:
                    frame = Image.fromarray(frame)
                    frame = self.transforms(frame)

            # Create a tensor for the intent with the same spatial dimensions as the video frames
            # Used for no intent
            intent_torch = torch.zeros((3, 128, 128))
            # If intent exists, add intent in its intent position for 1 second (10 frames)  
            if intent_position != -1 and intent_position <= i and (intent_position + 10) > i:
                # Convert intent to one-hot vector by filling the specified channel with 1
                intent_torch[intent, :, :] = 1

            # Append the intent as a channel to the video frame
            frame = torch.cat((frame, intent_torch), dim=0)
            frames.append(frame)
        cap.release()

        # Convert list to a 5D tensor [30, 6, 128, 128]
        video_tensor = torch.stack(frames, dim=0)
        
        # Memory cleanup: Delete intermediate list
        del frames, intent_torch

        return video_tensor, torch.tensor(label).long()
        
        # Proto 2: Simplified approach without intent 
        # csv_name = self.csv_files[idx]
        # csv_path = os.path.join(self.label_folder, csv_name)
        # 
        # df = pd.read_csv(csv_path)
        # label = self.labeler(df)
        #
        # cap = cv2.VideoCapture(video_path)
        # frames = []
        # for _ in range(30): 
        #     ret, frame = cap.read()
        #     if not ret:
        #         frame_tensor = torch.zeros((3, HEIGHT, WIDTH))
        #     else:
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         if self.transforms:
        #             frame = Image.fromarray(frame)
        #             frame_tensor = self.transforms(frame)
        #         else:
        #             frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        #     
        #     frames.append(frame_tensor)
        #     
        # cap.release()
        #
        # video_tensor = torch.stack(frames, dim=0)
        # del frames
        #
        # return video_tensor, torch.tensor(label).long()

    def labeler(self, df):
        # Original implementation
        df_lbl_count = []

        for i in range(0, 3):
            df_lbl_count.append(df['label_id_corrected'].tail(24).value_counts(i))

        if df_lbl_count[0] == 24:
            label = 0 # Front
        elif df_lbl_count[1] > df_lbl_count[2]:
            label = 1 # Left
        elif df_lbl_count[1] < df_lbl_count[2]:
            label = 2 # Right
        else: # If turn counts are equal
            label = df['label_id_corrected'].tail(12).mode()[0]

        return label
        
        # df_lbl_count = []
        # col = 'label_id_corrected' if 'label_id_corrected' in df.columns else df.columns[-1]
        # counts = df[col].tail(24).value_counts()
        # 
        # for i in range(0, 3):
        #     df_lbl_count.append(counts.get(i, 0))
        #
        # if df_lbl_count[0] == 24:
        #     label = 0  # Front
        # elif df_lbl_count[1] > df_lbl_count[2]:
        #     label = 1  # Left
        # elif df_lbl_count[1] < df_lbl_count[2]:
        #     label = 2  # Right
        # else: 
        #     label = df[col].tail(12).mode()[0]
        #
        # return label

    def get_intent_position(self):
        # 60% of the dataset have intent
        if random.random() < 0.6:
            # The time positions of the first 2 seconds (videos - 10 fps)
            start_frame = 0
            end_frame = 20
            median = (start_frame + end_frame)/2
            range_zero = np.arange(-median, median)

            # Obtain the probability of selecting a timestamp using the adjacent 0.5 areas
            smaller_range = range_zero - 0.5 
            higher_range = range_zero + 0.5    

            # Probability is the difference of the probability of higher range and lower range
            probability = ss.norm.cdf(higher_range) - ss.norm.cdf(smaller_range)
            
            # Normalize the probabilities
            # Each probability in probability range is divided by the sum of the probabilities in probability range
            probability /= probability.sum()

            # Select a timestamp based on the probabilities
            range = np.arange(start_frame, end_frame)
            intent_position = np.random.choice(range, p=probability)
        else:
            intent_position = -1
        
        return intent_position 

    def get_intent(self, intent_position, df):
        # Check if the data has no intent
        if intent_position != -1:
            intent = self.labeler(df)
        else:
            intent = -1
        return intent

    def class_counter(self):
        # Count instances of each class and sum of all class instances
        label_counts = {0: 0, 1: 0, 2: 0}
        
        for csv_file in self.csv_files:
            csv_path = os.path.join(self.label_folder, csv_file)
            df = pd.read_csv(csv_path)
            label = self.labeler(df)
            label_counts[label] += 1
        
        return label_counts, sum(label_counts.values())
        
        # Proto 2: Improved with CSV fallback (Eric's version - COMMENTED OUT)
        # label_counts = {0: 0, 1: 0, 2: 0}
        # for video_name in self.video_files:
        #     csv_name = video_name.replace('.mp4', '.csv')
        #     csv_path = os.path.join(self.label_folder, csv_name)
        #     
        #     if not os.path.exists(csv_path):
        #         csv_name = video_name.replace('.mp4', '_labels.csv')
        #         csv_path = os.path.join(self.label_folder, csv_name)
        #     
        #     if not os.path.exists(csv_path):
        #         csv_name = video_name.replace('.mp4', '_labels_A.csv')
        #         csv_path = os.path.join(self.label_folder, csv_name)
        #
        #     df = pd.read_csv(csv_path)
        #     label = self.labeler(df)        
        #     label_counts[label] += 1
        # return label_counts, sum(label_counts.values())
    
    def set_split_type(self, type, len_dataset):
        global VAL_POSITIONS
        global TEST_POSITIONS
        self.split_type = type

        if self.split_type == 'VALIDATION':
            if VAL_POSITIONS == '':
                for _ in range(len_dataset):
                    self.positions.append(self.get_intent_position())
                np.save('val_intent_positions.npy', np.array(self.positions))
                VAL_POSITIONS = 'val_intent_positions.npy'
            else:
                self.positions = list(np.load(VAL_POSITIONS))
        elif self.split_type == 'TEST':
            if TEST_POSITIONS == '':
                for _ in range(len_dataset):
                    self.positions.append(self.get_intent_position())
                np.save('test_intent_positions.npy', np.array(self.positions))
                TEST_POSITIONS = 'test_intent_positions.npy'
            else:
                self.positions = list(np.load(TEST_POSITIONS))

        return ''