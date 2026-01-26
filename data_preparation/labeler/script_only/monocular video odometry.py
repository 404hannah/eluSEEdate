import os
import numpy as np
import cv2
import csv
import math
from tqdm import tqdm
import ffmpeg
import datetime
import time
import shutil

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
        """
        Setting up the VO system. Loads calibration and opens video stream.
        """
        # Updated: Use script's actual directory instead of current working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calib_path = os.path.join(script_dir, 'calib.txt')
        
        self.K, self.P = self._load_calib(calib_path)
        
        # Streaming setup
        self.cap = cv2.VideoCapture(video_path)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.px_frame = None 
        
        self.orb = cv2.ORB_create(5000)
        
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

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Grayscale conversion for efficiency
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return None

    def detect_features(self, frame1, frame2):
        kp1, des1 = self.orb.detectAndCompute(frame1, None)
        kp2, des2 = self.orb.detectAndCompute(frame2, None)
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

def process_video(data_dir, vid_dir_name, video_file):
    video_path = os.path.join(data_dir, vid_dir_name, video_file)
    video_name = os.path.splitext(video_file)[0]
    labels_dir = os.path.join(data_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    vo = VisualOdometry(data_dir, video_path)
    csv_labels = []
    yaw_history = []
    cur_pose = np.eye(4)
    
    turn_threshold = 1.5
    window_size = 8

    for i in tqdm(range(vo.num_frames), desc=f"Video: {video_file}", unit="frame"):
        curr_frame = vo.get_next_frame()
        if curr_frame is None: break

        frame_id_str = f"{i:06d}.png"
        label_name = "FRONT"
        l_id = 0
        dist = 0.0
        smoothed_yaw = 0.0

        if i > 0 and vo.px_frame is not None:
            kp1, des1, kp2, des2 = vo.detect_features(vo.px_frame, curr_frame)
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
                

                if smoothed_yaw > (turn_threshold * turn_threshold):
                    l_id, label_name = 4, "RIGHT"
                elif smoothed_yaw > turn_threshold:
                    l_id, label_name = 3, "SLIGHT RIGHT"
                elif smoothed_yaw < -turn_threshold and smoothed_yaw > -(turn_threshold * turn_threshold):
                    l_id, label_name = 2, "SLIGHT LEFT"
                elif smoothed_yaw < -(turn_threshold * turn_threshold):
                    l_id, label_name = 1, "LEFT"
                else:
                    l_id, label_name = 0, "FRONT"
            else:
                label_name = "SKIPPED"

        csv_labels.append([frame_id_str, smoothed_yaw, dist, l_id, label_name])
        vo.px_frame = curr_frame

    vo.cap.release()

    # Saving Results to CSV
    csv_path = os.path.join(labels_dir, f"{video_name}_labels.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "yaw_degrees", "distance", "label_id", "label_name"])
        writer.writerows(csv_labels)

def fixing_outlier(labels_dir):
    # Locating and reading CSV files
    yaw_limit = 1.5 * 1.5 # Equal to the squared of turn_threshold 

    processed_lbl_folder = os.path.join(labels_dir, 'processed_labels') 
    os.makedirs(processed_lbl_folder, exist_ok=True)
    print(f"Verified folder: {processed_lbl_folder}")

    for csv_file in os.listdir(labels_dir):
        if csv_file.endswith("_labels.csv"):
            csv_path = os.path.join(labels_dir, csv_file)
            if csv_path == os.path.join(processed_lbl_folder, csv_file):
                continue
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                label_yaw = []
                label_ids = []
                label_yaw_corrected = []
                label_id_corrected = []
                label_name_corrected = []

                # Extract label_name and label_yaw columns
                for row in reader:
                    label_yaw.append(row['yaw_degrees'])
                    label_ids.append(row['label_id'])

            counter = 0
            # Identifying and correcting outliers
            for i, ids in enumerate(label_ids):
                if i == 0 or i == len(label_ids) - 1:
                    label_id_corrected.append(ids)
                    label_yaw_corrected.append(float(label_yaw[i])) 
                    continue
                # An outlier is detected if it is dissimilar from both neighbors
                if label_id_corrected[i-1] == label_ids[i+1] and ids != label_id_corrected[i-1]:
                    # Keep lone outlier with high yaw_degrees
                    if float(label_yaw[i]) > yaw_limit:
                        label_yaw[i+1] = float(label_yaw[i+1]) + (float(label_yaw[i]) - yaw_limit)
                        label_yaw_corrected.append(float(label_yaw[i]) - ((float(label_yaw[i]) - yaw_limit))) 
                        label_id_corrected.append(ids)
                    elif float(label_yaw[i]) < -(yaw_limit):
                        label_yaw[i+1] = float(label_yaw[i+1]) + (float(label_yaw[i]) + 2.25)
                        label_yaw_corrected.append(float(label_yaw[i]) - ((float(label_yaw[i]) + yaw_limit))) 
                        label_id_corrected.append(ids)
                    else: 
                        label_id_corrected.append(label_ids[i-1]) # Correct the outlier   
                        label_yaw_corrected.append(float(label_yaw[i]))
                        counter += 1
                else:
                    label_id_corrected.append(ids)
                    label_yaw_corrected.append(float(label_yaw[i]))

            print(f"{os.path.splitext(csv_file)[0]}.csv -> No. of outliers: {counter}")

            # Providing the corresponding corrected label names
            for label in label_id_corrected:
                if label == '0':
                    label_name_corrected.append('FRONT')
                elif label == '1':
                    label_name_corrected.append('LEFT')
                elif label == '2':
                    label_name_corrected.append('SLIGHT LEFT')
                elif label == '3':
                    label_name_corrected.append('SLIGHT RIGHT')
                elif label == '4':
                    label_name_corrected.append('RIGHT')
                else:
                    label_name_corrected.append('SKIPPED')

            # Read the file again from the beginning
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Now write the corrected data
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                for i, row in enumerate(rows):
                    # Write header row with new columns
                    if i == 0:
                        row.append('label_yaw_corrected')
                        row.append('label_id_corrected')
                        row.append('label_name_corrected')
                        writer.writerow(row)
                    # Add the modified values in each row of the CSV file
                    else:
                        row.append(label_yaw_corrected[i-1])
                        row.append(label_id_corrected[i-1])
                        row.append(label_name_corrected[i-1])
                        writer.writerow(row)

            # All processed csv files are copied to a folder 
            shutil.copy(csv_path, os.path.join(processed_lbl_folder, csv_file))         
    return

def main():
    start = time.time()
    print("-" * 30)
    print("Starting Monocular Visual Odometry (Headless Mode)")
    print("-" * 30)     
    
    # Input path for the videos folder
    video_folder = r''
    vid_dir_name = os.path.basename(video_folder)
    
    data_dir = os.path.dirname(video_folder)
    label_folder = os.path.join(data_dir, 'labels')

    os.makedirs(label_folder, exist_ok=True)
    print(f"Verified folder: {label_folder}")
    
    input_size_mb = get_folder_size(video_folder) 
    formatted_original_time, total_input_seconds, input_file_count = get_video_stats(video_folder)
    
    print(f"Total duration: {formatted_original_time}")
    print(f"Total size: {input_size_mb} MB")
    print("-" * 30)     

    if not os.path.exists(video_folder):
        print(f"Error: {video_folder} not found.")
        return

    extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(extensions)]
    
    for video_file in video_files:
        process_video(data_dir, vid_dir_name, video_file)

    fixing_outlier(os.path.join(data_dir, "labels"))
    
    print("\n" + "-" * 30)     
    print(f"Total duration of source videos: {formatted_original_time}")
    print(f"Processing finished for all {len(video_files)} videos")    
    print(f"Total size of source videos: {input_size_mb} MB")    
    
    end = time.time()
    print(f"\nProcessing time: {round(end-start, 2)} seconds")
    
    print("-" * 30)   

if __name__ == "__main__":
    main()