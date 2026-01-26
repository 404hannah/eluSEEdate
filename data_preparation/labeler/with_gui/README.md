# 🚗 Visual Odometry & Turn Predictor

A Python-based **Visual Odometry (VO)** system that estimates a vehicle's trajectory from a single camera feed and predicts driving directions (**LEFT**, **RIGHT**, or **FRONT**) in real-time using geometric computer vision.

<p>
  <img alt="Python" src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  
  <img alt="NumPy" src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img alt="OpenCV" src="https://img.shields.io/badge/-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img alt="FFmpeg" src="https://img.shields.io/badge/-FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white" />
  <img alt="tqdm" src="https://img.shields.io/badge/-tqdm-FFC107?style=for-the-badge&logo=python&logoColor=black" />
  
  <img alt="os" src="https://img.shields.io/badge/-os-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
  <img alt="csv" src="https://img.shields.io/badge/-csv-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
  <img alt="math" src="https://img.shields.io/badge/-math-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
  <img alt="datetime" src="https://img.shields.io/badge/-datetime-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
  <img alt="time" src="https://img.shields.io/badge/-time-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
</p>


## 🌟 Features
* **Real-time Feature Matching:** Uses ORB descriptors and FLANN matching to track landmarks between frames.
* **Trajectory Estimation:** Calculates relative rotation and translation to map the vehicle's path.
* **Turn Prediction:** Uses a moving average of the camera's yaw to categorize movements into driving labels.
* **Vertical GUI Layout:** A customized "Matches" window stacked vertically for better visibility on standard monitors.
* **Clean Data Export:** Automatically organizes output into `labels/` (CSV) and `bokeh/` (Interactive HTML maps).

---

## 📂 Project Structure
```text
.
├── Datasets/
│   ├── videos/          # Raw input video files (.mp4, .avi, etc.)
│   ├── labels/          # Auto-generated CSV label files
│   └── calib.txt        # Camera calibration parameters (P-matrix)
├── lib/visualization
│   ├── camera.py        # Provides functions to visualize and plot the 3D position and orientation
│   ├── image.py         # Contains utility functions for drawing text labels and matching-point lines on images to help visualize the output of computer vision algorithms
│   └── video.py         # Acts as a simple playback engine that loops through image frames to simulate a video feed for debugging purposes
└── visual_odometry.py   # Main processing script
```

## 🚀 How It Works

The system follows a standard Monocular Visual Odometry pipeline:
* Feature Detection: ORB looks for stable "landmarks" in the image.
* Feature Matching: FLANN connects points from the previous frame to the current frame.
* Pose Estimation: The Essential Matrix is calculated via RANSAC to determine how the camera moved.
* Prediction Logic
  * FRONT: Yaw rotation is within ±1.25∘.
  * LEFT: Yaw rotation is less than −1.25∘.
  * RIGHT: Yaw rotation is greater than 1.25∘
 
## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ensues/Monocular-VO-for-Automated-Turn-Labeling-and-CSV-Generation.git
   cd Monocular-VO-for-Automated-Turn-Labeling-and-CSV-Generation
2. Prepare Data:
   * Place your video files in Datasets/videos/.
   * Ensure Datasets/calib.txt contains your camera matrix, which is your camera focal point expressed in a matrix, the default is a general one for 480 resolution videos.
3. Run the Project:
    ```text
    python visual_odometry.py
    ```

## 📊 Outputs
### CSV Labels
Saves high-fidelity data for training or analysis:
  * frame_id: The filename/ID of the frame.

  * yaw_degrees: The smoothed rotation value.
 
  * distance: Movement in meters (unitless scale).

  * label_id: Numerical ID (0: Front, 1: Left, 2: Right).

## ⚙️ Configuration
You can fine-tune the system in visual_odometry.py:
  * turn_threshold: Sensitivity for triggering a "Turn" label.
  * window_size: The number of frames used to smooth out the yaw.
  * cv2.resizeWindow: Changes the GUI size (Default: 920x920).