import ffmpeg
import os
import datetime
import time
import pandas as pd
import shutil
from pathlib import Path

# Functions cuz I can
def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    # Returns size in MB for readability
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
                # Skip files that are corrupted or not readable by ffmpeg
                pass
    
    formatted_time = str(datetime.timedelta(seconds=int(total_seconds)))
    return formatted_time, file_count

def inspect_csv_folder(directory_path):
    csv_stats = []
    csv_count = 0

    # Ensures the directory exists
    if not os.path.exists(directory_path):
        return "Path not found. Double-check your folder string!"

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            csv_count += 1
            file_path = os.path.join(directory_path, filename)
            
            try:
                # We use pandas to quickly get the shape (rows, columns)
                df = pd.read_csv(file_path)
                rows, cols = df.shape
                csv_stats.append({
                    "file": filename,
                    "rows": rows,
                    "columns": cols
                })
            except Exception as e:
                csv_stats.append({"file": filename, "error": str(e)})

    return {
        "total_csvs_found": csv_count,
        "details": csv_stats
    }
    
# Define I/O folders

input_augmented_folder = r'' 
input_csv_folder = r''

parent_folder = os.path.dirname(input_augmented_folder)
output_folder = os.path.join(parent_folder, 'Full Augmented Videos')
video_out = os.path.join(output_folder, "augmented_videos")
csv_out = os.path.join(output_folder, "augmented_csvs")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(video_out, exist_ok=True)
os.makedirs(csv_out, exist_ok=True)

print("-" * 30)

# Calculating the total duration

print("Calculating total duration and size of input videos...")

vid_input_size_mb = get_folder_size(input_augmented_folder) 
formatted_original_time, input_file_count = get_video_stats(input_augmented_folder)

csv_input_size_mb = get_folder_size(input_csv_folder) 

input_csv_results = inspect_csv_folder(input_csv_folder)
print(input_csv_results)

print(f"Total duration of all source videos: {formatted_original_time}")
print(f"Total size of input folder: {vid_input_size_mb} MB")

print("-" * 30)

start = time.time()

csv_map = {Path(f).stem.removesuffix('_labels'): f for f in os.listdir(input_csv_folder) if f.endswith('.csv')}

video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
match_count = 0

for vid_filename in os.listdir(input_augmented_folder):
    if vid_filename.lower().endswith(video_extensions):
        vid_stem = Path(vid_filename).stem
        
        # Check if there is a matching CSV file
        if vid_stem in csv_map:
            csv_filename = csv_map[vid_stem]
            
            # Create the new names with the "_A" postfix
            new_vid_name = vid_filename + "_A" 
            new_csv_name = csv_filename + "_A" 
            
            # Define full paths
            src_vid = os.path.join(input_augmented_folder, vid_filename)
            dst_vid = os.path.join(video_out, new_vid_name)
            
            src_csv = os.path.join(input_csv_folder, csv_filename)
            dst_csv = os.path.join(csv_out, new_csv_name)
            
            # Copy the files
            shutil.copy2(src_vid, dst_vid)
            shutil.copy2(src_csv, dst_csv)
            
            print(f"✔ Processed: {vid_stem}")
            match_count += 1

end = time.time()

# Extras

print("-" * 30)
print(f"Done! Pairs copied to '{output_folder}': {match_count}")

print("-" * 30)
print("Calculating final dataset duration...")

output_size_augmented_mb = get_folder_size(video_out)
formatted_final_augmented_time, output_file_augmented_count = get_video_stats(video_out)

final_csv_results = inspect_csv_folder(csv_out)

print(f"Processing time: {round(end-start, 2)} seconds or {round((end-start)/60, 2)} minutes\n")

print(f"Total duration of all videos: {formatted_final_augmented_time}")
print(f"Total videos: {output_file_augmented_count}\n")

print(f"Total size of input folder: {vid_input_size_mb} MB")
print(f"Total size of output folder: {output_size_augmented_mb} MB")

print(final_csv_results)

print("-" * 30)