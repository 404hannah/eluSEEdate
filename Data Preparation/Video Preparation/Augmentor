# LIBRARIES

import ffmpeg
import os
import datetime
import random
import time

# Functions

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


# FOLDER PREPARATION

# Define input folder

input_folder = r'D:\Thesis 2\Input'
parent_folder = os.path.dirname(input_folder)
augmented_main_folder = os.path.join(parent_folder, 'Augmented Dataset Videos')

# Define output folders

output_1variant_folder = os.path.join(augmented_main_folder, '1Variant')
output_2variants_folder = os.path.join(augmented_main_folder, '2Variants')
output_final_folder = os.path.join(augmented_main_folder, 'Augmented Data')

# List to check

output_folders = [
    output_1variant_folder, 
    output_2variants_folder, 
    output_final_folder,
]

# Create the output folder if it doesn't exist

os.makedirs(augmented_main_folder, exist_ok=True)
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Verified folder: {folder}")

# Calculating the total duration

print("-" * 30)

print("Calculating total duration and size of input videos...")

input_size_mb = get_folder_size(input_folder) 
formatted_original_time, total_input_seconds, input_file_count = get_video_stats(input_folder)

print(f"Total duration of all source videos: {formatted_original_time}")
print(f"Total size of input folder: {input_size_mb} MB")

print("-" * 30)


# DATA AUGMENTATION

# Limitations
processed_count = 0
total_augmented_seconds = 0

augmentations = [
    {'name': 'Brighter', 'vf': 'eq=brightness=0.3'},
    {'name': 'Dimmer', 'vf': 'eq=brightness=-0.3'},
    # noise=Luminance=Strength:Function=Uniform
        # Luminance for the brightness
        # Strength for the intensity
        # Uniform to make the noise brighter or darker
    {'name': 'Noise', 'vf': 'noise=c0s=50:c0f=t+u'},
    # pad=iw+5:ih+5:5:5: Adds 5 pixels of padding to the top and left
    {'name': 'Translation', 'vf': 'setpts=PTS-STARTPTS,pad=iw+5:ih+5:5:5:black'},
    # pixelize=width=16:height=16: Tells FFmpeg to divide the image into 16×16 pixel blocks
    {'name': 'Superpixel', 'vf': 'pixelize=width=16:height=16'}
]

start = time.time()
for filename in os.listdir(input_folder):

    # A video has a 30% chance of undergoing augmentation
    if filename.lower().endswith(".mp4") and random.randrange(1, 101) <= 30:
        input_path = os.path.join(input_folder, filename)
        processed_count += 1
        
        # Calculate augmented video time
        try:
            probe = ffmpeg.probe(input_path)
            vid_duration = float(probe['format']['duration'])
            # Since we create 3 variants per chosen video, we add it 3 times
            total_augmented_seconds += (vid_duration * 3)
        except:
            vid_duration = 0
        
        print(f"\nProcessing Video {processed_count}: {filename}")
        variants = []
        
        # Three distinct augmentation occurs per video
        for i in range(len(output_folders)):
            aug = random.choice(augmentations)
            # Repeatedly chooses a variant if the generated variant is chosen previously
            # Only one between brightness and dimmer could be chosen
            while aug in variants or (aug['name'] == 'Brighter' and {'name': 'Dimmer', 'vf': 'eq=brightness=-0.3'} in variants) or (aug['name'] == 'Dimmer' and {'name': 'Brighter', 'vf': 'eq=brightness=0.3'} in variants):
                aug = random.choice(augmentations)

            variants.append(aug)

            current_var = variants[len(variants)-1]
            output_path = os.path.join(output_folders[i], filename)

            # Check if this specific augmentation already exists
            if os.path.exists(output_path):
                print(f"  > Skipping Augmentation {i}: Already exists")
                continue
                
            try:
                # Logic: Only apply the specific augmentation filter
                (
                    ffmpeg
                    .input(input_path if i == 0 else os.path.join(output_folders[i-1], filename))
                    .output(output_path, vf=current_var['vf']) 
                    .overwrite_output()
                    .run(quiet=True)
                )
                print(f"  > Done: {current_var['name']}")
            except ffmpeg.Error as e:
                print(f"  ! Error on {current_var['name']}: {e}")

# FINISHING TOUCHES

end = time.time()
formatted_augmented_time = str(datetime.timedelta(seconds=int(total_augmented_seconds)))

output_size_mb = get_folder_size(output_final_folder) 
formatted_augmented_time, total_output_seconds, output_file_count = get_video_stats(output_final_folder)

print("-" * 30)
print("Videos have been augmented")
print(f"Processing time: {round(end-start, 2)} seconds")

print(f"\nTotal duration of all source videos: {formatted_original_time}")
print(f"Total duration of augmented videos: {formatted_augmented_time}")

print(f"\nTotal source videos: {input_file_count}")
print(f"Total augmented videos: {output_file_count}")

print(f"\nTotal size of input folder: {input_size_mb} MB")
print(f"Total size of augmented main folder: {output_size_mb} MB")
print("-" * 30)

# test