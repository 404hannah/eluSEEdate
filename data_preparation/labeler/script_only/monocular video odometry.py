import os
import numpy as np
import cv2
import csv
import math
from tqdm import tqdm
import ffmpeg
import datetime
import time

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)

def get_video_stats(folder_path):
    total_seconds = 0
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".mp4"):
            path = os.path.join(folder_path, filename)
            try:
                probe = ffmpeg.probe(path)
                duration = float(probe['format']['duration'])
                total_seconds += duration
                file_count += 1
            except Exception:
                pass
    formatted_time = str(datetime.timedelta(seconds=int(total_seconds)))
    return formatted_time, total_seconds, file_count

class VisualOdometry():
    def __init__(self, data_dir, video_path):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.orb = cv2.ORB_create(3000)
        self.prev_frame = None 
        
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=dict(checks=50))

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    def _form_transf(self, R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def detect_features(self, current_frame):
        # Match previous frame buffer against the current frame
        kp1, des1 = self.orb.detectAndCompute(self.prev_frame, None)
        kp2, des2 = self.orb.detectAndCompute(current_frame, None)
        return kp1, des1, kp2, des2

    def match_features(self, des1, des2):
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        return good

    def get_pose(self, q1, q2):
        E, mask = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None or E.shape != (3,3): 
            return np.eye(4)
        
        R, t = self.decomp_essential_mat(E, q1[mask.ravel()==1], q2[mask.ravel()==1])
        return self._form_transf(R, np.squeeze(t))

    def decomp_essential_mat(self, E, q1, q2):
        def sum_z(R, t):
            T = self._form_transf(R, t)
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                            (np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1) + 1e-8))
            return sum(uhom_Q1[2, :] > 0) + sum(uhom_Q2[2, :] > 0), scale

        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        results = [sum_z(R, tr) for R, tr in [[R1, t], [R1, -t], [R2, t], [R2, -t]]]
        
        idx = np.argmax([r[0] for r in results])
        best_pair = [[R1, t], [R1, -t], [R2, t], [R2, -t]][idx]
        return best_pair[0], best_pair[1] * results[idx][1]

    def get_yaw(self, R):
        return -math.degrees(math.atan2(R[0, 2], R[2, 2]))

def process_video(data_dir, video_file):
    video_path = os.path.join(data_dir, "videos", video_file)
    video_name = os.path.splitext(video_file)[0]
    labels_dir = os.path.join(data_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    vo = VisualOdometry(data_dir, video_path)
    csv_labels = []
    yaw_history = []
    cur_pose = np.eye(4)
    
    turn_threshold = 1.25
    window_size = 5

    for i in tqdm(range(vo.total_frames), desc=f"Processing: {video_file}", unit="frame"):
        ret, frame = vo.cap.read()
        if not ret:
            break
            
        # Convert to grayscale for processing
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i == 0:
            vo.prev_frame = current_gray
            csv_labels.append([f"{i:06d}.png", 0.0, 0.0, 0, "FRONT"])
            continue

        frame_id_str = f"{i:06d}.png"
        label_name = "FRONT"
        l_id, dist, smoothed_yaw = 0, 0.0, 0.0

        kp1, des1, kp2, des2 = vo.detect_features(current_gray)
        good_matches = vo.match_features(des1, des2)
        
        if len(good_matches) > 8:
            q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            
            dist = np.linalg.norm(transf[:3, 3])
            raw_yaw = vo.get_yaw(transf[:3, :3])
            yaw_history.append(raw_yaw)
            if len(yaw_history) > window_size: yaw_history.pop(0)
            smoothed_yaw = sum(yaw_history) / len(yaw_history)
            
            if smoothed_yaw > turn_threshold:
                l_id, label_name = 2, "RIGHT"
            elif smoothed_yaw < -turn_threshold:
                l_id, label_name = 1, "LEFT"
        else:
            label_name = "SKIPPED"

        csv_labels.append([frame_id_str, smoothed_yaw, dist, l_id, label_name])
        
        vo.prev_frame = current_gray

    vo.cap.release()

    csv_path = os.path.join(labels_dir, f"{video_name}_labels.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "yaw_degrees", "distance", "label_id", "label_name"])
        writer.writerows(csv_labels)

def main():
    start = time.time()
    
    print("-" * 30)
    print("Starting Monocular Visual Odometry")
    print("-" * 30)     
    
    data_dir = "datasets"
    video_folder = os.path.join(data_dir, "videos")
    
    input_size_mb = get_folder_size(video_folder) 
    formatted_original_time, total_input_seconds, input_file_count = get_video_stats(video_folder)
    
    print(f"Total duration of all source videos: {formatted_original_time}")
    print(f"Total count of source videos: {input_file_count}")
    print(f"Total size of source folder: {input_size_mb} MB")
    print("-" * 30)   
    
    if not os.path.exists(video_folder):
        print(f"Error: {video_folder} not found.")
        return

    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        process_video(data_dir, video_file)
        
    print(f"\nProcessing finished for all {input_file_count} videos")        
    print("-" * 30)   

    end = time.time()
    print(f"Processing time: {round(end-start, 2)} seconds")
    print("-" * 30)

if __name__ == "__main__":
    main()