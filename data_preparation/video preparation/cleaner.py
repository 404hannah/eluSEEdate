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

input_folder = r''
parent_folder = os.path.dirname(input_folder)
output_folder = os.path.join(parent_folder, 'Cleaned Dataset Videos')

# Create the output folder if it doesn't exist

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("-" * 30)

# Calculating the total duration

print("Calculating total duration and size of input videos...")

input_size_mb = get_folder_size(input_folder) 
formatted_original_time, total_input_seconds, input_file_count = get_video_stats(input_folder)

print(f"Total duration of all source videos: {formatted_original_time}")
print(f"Total size of input folder: {input_size_mb} MB")

print("-" * 30)

# Loop through all files in the input folder

print(f"Starting video cleaning\n")

start = time.time()

global_file_counter = 1

for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        # Each file is saved as a unique numbered video
        output_path = os.path.join(output_folder, f"{global_file_counter:010d}.mp4")
        
        # Shows file being processed
        print(f"Processing: {filename} (Saving as {global_file_counter:010d}.mp4)...")

        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path, 
                    # Scale
                    vf='scale=128:128',
                    # Grayscale and Scale
                    # vf='format=gray,scale=64:64', 
                    # Framerate
                    r=10,
                    # Disable audio
                    an=None, 
                    # Remove rotation metadata
                    metadata='s:v:0 rotate=0'
                )
                .overwrite_output() # Overwrites if file exists
                .run(quiet=True)
            )
            
            global_file_counter += 1
            
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