import ffmpeg
import os
import datetime
import time

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
    return formatted_time, total_seconds, file_count

# Define I/O folders

input_folder = r'' # Remember to change author
parent_folder = os.path.dirname(input_folder)
output_folder = os.path.join(parent_folder, 'Segmented Dataset Videos')

# Create the output folder if it doesn't exist

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Change for first letter of the name of who is using this
author = "E"

print("-" * 30)

# Calculating the total duration

print("Calculating total duration and size of input videos...")

input_size_mb = get_folder_size(input_folder) 
formatted_original_time, total_input_seconds, input_file_count = get_video_stats(input_folder)

print(f"Total duration of all source videos: {formatted_original_time}")
print(f"Total size of input folder: {input_size_mb} MB")

print("-" * 30)

# Loop through all files in the input folder

print(f"Starting video cleaning and segmentation\n")

start = time.time()

global_segment_counter = 1

for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        output_template = os.path.join(output_folder, author + "%010d.mp4")
        
        # Shows file being processed
        print(f"Processing: {filename} (Starting at index {global_segment_counter:010d})...")

        try:
            probe = ffmpeg.probe(input_path)
            duration = float(probe['format']['duration'])
            num_segments = int(duration // 3) + (1 if duration % 3 > 0 else 0)
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_template, 
                    # Segment logic: Split every 3 seconds
                    f='segment',
                    segment_time=3,
                    reset_timestamps=1,
                    # Start count of the specific vid
                    segment_start_number=global_segment_counter,
                    force_key_frames='expr:gte(t,n_forced*3)'
                    )
                .overwrite_output() # Overwrites if file exists
                .run(quiet=True)
            )
            
            # Cleanup Logic
            segments_actually_kept = 0
            for i in range(global_segment_counter, global_segment_counter + num_segments):
                seg_path = os.path.join(output_folder, f"{author}{i:010d}.mp4")
                if os.path.exists(seg_path):
                    try:
                        seg_probe = ffmpeg.probe(seg_path)
                        seg_dur = float(seg_probe['format']['duration'])
                        # If duration is less than 2.9s, it's a "leftover" fragment
                        if seg_dur < 2.9: 
                            os.remove(seg_path)
                        else:
                            segments_actually_kept += 1
                    except Exception:
                        # If the file is corrupted/0kb, probe fails, so we delete it
                        os.remove(seg_path)
            
            global_segment_counter += segments_actually_kept
        except ffmpeg.Error as e:
            print(f"Error processing {filename}: {e.stderr.decode() if e.stderr else 'Unknown error'}")

end = time.time()

# Extras

print("-" * 30)
print("Calculating final dataset duration...")

output_size_mb = get_folder_size(output_folder)
formatted_final_time, total_output_seconds, output_file_count = get_video_stats(output_folder)

print("-" * 30)
print("All videos have been processed\n")

print(f"Processing time: {round(end-start, 2)} seconds\n")

print(f"Total duration of all source videos: {formatted_original_time}")
print(f"Total raw videos: {input_file_count}")
print(f"Total size of input folder: {input_size_mb} MB\n")

print(f"Total duration of all cleaned videos: {formatted_final_time}")
print(f"Total processed videos: {output_file_count}")
print(f"Total size of output folder: {output_size_mb} MB")
print("-" * 30)