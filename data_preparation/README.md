# 🛠️ Data Preparation & Labeling Pipeline

This repository contains a specialized pipeline designed for processing raw video data, applying augmentations, and generating ground-truth labels using Monocular Visual Odometry (MVO).

<p>
  <img alt="Python" src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" /> <img alt="NumPy" src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" /> <img alt="OpenCV" src="https://img.shields.io/badge/-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" /> <img alt="FFmpeg" src="https://img.shields.io/badge/-FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white" /> <img alt="tqdm" src="https://img.shields.io/badge/-tqdm-FFC107?style=for-the-badge&logo=python&logoColor=black" /> <img alt="os" src="https://img.shields.io/badge/-os-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> <img alt="csv" src="https://img.shields.io/badge/-csv-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> <img alt="math" src="https://img.shields.io/badge/-math-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> <img alt="datetime" src="https://img.shields.io/badge/-datetime-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> <img alt="time" src="https://img.shields.io/badge/-time-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> 
</p>

## 📂 Project Structure

All assets are contained within the data_preparation root folder, organized into two primary sub-directories:

```text
data_preparation/
├── balancer/
│   └── balancer.py          # Step 5: Majority undersampling
├── csv preparation/
│   └── csv migrator.py      # Step 8: Duplicates CSVs for augmented data
│   └── outlier remover.py    # Step 4: Removes outliers
├── labeler/
│   ├── MVO_script.py        # Step 3: Fast labeling without visualization
│   └── MVO_gui.py           # Step 3: Visual labeling with matching preview
└──video preparation/
    ├── augmentor.py         # Step 7: Diverse visual variations
    ├── cleaner.py           # Step 6: Final cleanup
    ├── rescaler.py          # Step 1: Standardize resolution/FPS
    └── segmenter            # Step 2: Segments the videos to 3 sec durations
```

## 🚀 Workflow Execution Order

To ensure data integrity and proper formatting, scripts must be executed in the following order:

1. **rescaler**: Standardizes all input videos to a manageable resolution (480p) and fixed frame rate.
2. **segmenter**: Segments the labeled videos into consistent 3-second clips for model training.
3. **MVO**: Generates turn labels (Left, Slight Left, Slight Right, Front) using Visual Odometry.
4. **outlier remover**: Removes class instance by copying the neighbor if both neighbors are the same.
5. **balancer**: Reduces class imbalance by removing data with all front classes.
6. **cleaner**: Scales the videos to 128x128 and 10 fps for model training.
7. **augmentor**: Generates variants (Brightness, Noise, etc.) to increase dataset diversity.
8. **csv migrator**: Duplicates CSVs for augmented videos

## 🎥 Video Preparation Tools

1. **Rescaler** (rescaler.py)
   Standardizes raw footage.

- Scale: -1:480 (maintains aspect ratio).
- FPS: 24.
- Action: Removes audio and clears rotation metadata to prevent orientation errors.

2. **Segmenter** (segmenter.py)
   Prepares the videos for the MVO

- Segmentation: Splits videos into exact 3-second segments.
- Cleanup: Automatically deletes "leftover" fragments shorter than 2.9 seconds.

3. **Cleaner** (cleaner.py)
   The final processing step.

- Scaling: Downscales to 128x128 for neural network input.
- Frame rate reduction: Reduced to 10fps for neural network input.

4. **Augmentor** (augmentor.py)
   Applies a 30% probability of augmentation to videos, creating three distinct variants:

- Brighter/Dimmer: Simulates lighting changes.
- Noise: Adds luminance-based grain.
- Translation: Shifts the frame 5 pixels.
- Superpixel: Applies 16x16 block pixelization.

## 🏷️ Labeler (Monocular Visual Odometry)

The labeling tool uses geometric computer vision to track camera movement and predict turns based on Yaw rotation.

    [!IMPORTANT] Both labeling scripts require a calib.txt file in the dataset directory containing your camera's projection matrix.

Choose Your Mode:

- **Script Only**: Optimized for speed. Processes videos in the background and exports CSVs directly to the labels/ folder. Use this for large-scale batch processing.
- **With GUI**: Best for debugging or verification. Provides a real-time window showing:
  - Feature Matching: ORB landmarks being tracked between frames.
  - Trajectory: Real-time turn prediction (LEFT/RIGHT/FRONT).
  - Visual Feedback: Green/Red lines indicating match quality.

Prediction Logic
| Direction | Yaw Threshold | Label ID |
|----------|----------|----------|
| FRONT | Within ±1.5∘ | 0 |
| LEFT | <−1.5∘ | 1 |
| RIGHT | >1.5∘ | 2 |

## ⚖️ Balancer

This script uses majority undersampling to reduce the class imbalance. It also calculates the instances of each class.

## 🛠️ Setup & Usage

1. Dependencies: Ensure you have Python 3.x installed with the libraries listed in the badges above.

2. Initialize Video Folder Paths:

Labeler and Balancer
| Script | monocular visual odometry.py | balancer.py | csv migrator.py | outlier remover.py |
| -------- | -------- | -------- | -------- | -------- |
| Variable | `video_folder = r''` | `label_folder = r'' video_folder = r'' ` | `input_augmented_folder = r'' input_csv_folder = r''` | `labels_dir = r''`
| Directory | labeler/script_only/ | balancer/ | labeler/csv preparation/ | labeler/csv preparation/ |

Video preparation
| Script | augmentor.py | cleaner.py | rescaler.py | segmenter.py |
| -------- | -------- | -------- | -------- | -------- |
| Variable | `video_folder = r''` | `input_folder = r''` | `input_folder = r''` | `input_folder = r''` | `input_folder = r''` |
| Directory | video preparation/ | video preparation/ | video preparation/ | video preparation/ |

3. video preparation/Input author initial:

In segmenter.py <br>
`author = ""`

4. Run Pipeline:

```text
# Example Step 1
python video_preparation/rescaler.py
```
