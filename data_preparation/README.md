# 🛠️ Data Preparation & Labeling Pipeline

This repository contains a specialized pipeline designed for processing raw video data, applying augmentations, and generating ground-truth labels using Monocular Visual Odometry (MVO).

<p>
  <img alt="Python" src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" /> <img alt="NumPy" src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" /> <img alt="OpenCV" src="https://img.shields.io/badge/-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" /> <img alt="FFmpeg" src="https://img.shields.io/badge/-FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white" /> <img alt="tqdm" src="https://img.shields.io/badge/-tqdm-FFC107?style=for-the-badge&logo=python&logoColor=black" /> <img alt="os" src="https://img.shields.io/badge/-os-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> <img alt="csv" src="https://img.shields.io/badge/-csv-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> <img alt="math" src="https://img.shields.io/badge/-math-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> <img alt="datetime" src="https://img.shields.io/badge/-datetime-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> <img alt="time" src="https://img.shields.io/badge/-time-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> 
</p>

## 📂 Project Structure

All assets are contained within the data_preparation root folder, organized into two primary sub-directories:

```text
data_preparation/
├── video_preparation/
│   ├── augmentor.py     # Step 2: Diverse visual variations 
│   ├── cleaner.py       # Step 5: Final cleanup
│   ├── rescaler.py      # Step 1: Standardize resolution/FPS
│   └── segmenter        # Step 3: Segments the videos to 3 sec durations
└── labeler/
    ├── MVO_script.py    # Step 4: Fast labeling without visualization
    └── MVO_gui.py       # Step 4: Visual labeling with matching preview
```

## 🚀 Workflow Execution Order

To ensure data integrity and proper formatting, scripts must be executed in the following order:
1. **rescaler**: Standardizes all input videos to a manageable resolution (480p) and fixed frame rate.
2. **augmentor**: Generates variants (Brightness, Noise, etc.) to increase dataset diversity.
3. **segmenter**: Segments the labeled videos into consistent 3-second clips for model training.
4. **MVO**: Generates turn labels (Left, Right, Front) using Visual Odometry.
5. **cleaner**: Scales the videos to 128x128 and 10 fps for model training.

## 🎥 Video Preparation Tools
1. **Rescaler** (rescaler.py)
Standardizes raw footage.
* Scale: -1:480 (maintains aspect ratio).
* FPS: 24.
* Action: Removes audio and clears rotation metadata to prevent orientation errors.

2. **Augmentor** (augmentor.py)
Applies a 30% probability of augmentation to videos, creating three distinct variants:
* Brighter/Dimmer: Simulates lighting changes.
* Noise: Adds luminance-based grain.
* Translation: Shifts the frame 5 pixels.
* Superpixel: Applies 16x16 block pixelization.

3. **Segmenter** (segmenter.py)
Prepares the videos for the MVO
* Segmentation: Splits videos into exact 3-second segments.
* Cleanup: Automatically deletes "leftover" fragments shorter than 2.9 seconds.

4. **Cleaner** (sequencer.py)
The final processing step.
* Scaling: Downscales to 64×64 for neural network input.
* Frame rate reduction: Reduced to 10fps for neural network input.


## 🏷️ Labeler (Monocular Visual Odometry)
The labeling tool uses geometric computer vision to track camera movement and predict turns based on Yaw rotation.

    [!IMPORTANT] Both labeling scripts require a calib.txt file in the dataset directory containing your camera's projection matrix.

Choose Your Mode:
* **Script Only**: Optimized for speed. Processes videos in the background and exports CSVs directly to the labels/ folder. Use this for large-scale batch processing.
* **With GUI**: Best for debugging or verification. Provides a real-time window showing:
  * Feature Matching: ORB landmarks being tracked between frames.
  * Trajectory: Real-time turn prediction (LEFT/RIGHT/FRONT).
  * Visual Feedback: Green/Red lines indicating match quality.

Prediction Logic
| Direction | Yaw Threshold | Label ID |
|----------|----------|----------|
| FRONT | Within ±1.25∘ | 0 |
| LEFT | <−1.25∘ | 1 |
| RIGHT | >1.25∘ | 2 |

## 🛠️ Setup & Usage

To be updated pa

1. Dependencies: Ensure you have Python 3.x installed with the libraries listed in the badges above.
2. Initialize Folders:
  * Place raw videos in Datasets/videos/.
  * Ensure Datasets/calib.txt is present.
3. Run Pipeline:
```text
# Example Step 1
python video_preparation/rescaler.py
```
