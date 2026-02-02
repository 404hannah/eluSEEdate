import os
import csv
import shutil
import time

def class_counter(label_folder, threshold):
    total_count = {'front': 0, 'left': 0, 'right': 0}
    front_csv = []
    file_count = 0
    pred_removed = {'front': 0, 'left': 0, 'right': 0}

    # Counts the occurrences of each class label in the given folder.
    print(f"Counting instances of each class from the csv files.")

    for file in os.listdir(label_folder):
        if file.endswith(".csv"):
            csv_path = os.path.join(label_folder, file)
            in_file_count = {'front': 0, 'left': 0, 'right': 0}
            row_count = 0

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)

                # Extract label_id_corrected values
                for row in reader:
                    row_count += 1
                    if int(row['label_id_corrected']) == 0:
                        total_count['front'] += 1
                        in_file_count['front'] += 1
                    elif int(row['label_id_corrected']) == 1:
                        total_count['left'] += 1
                        in_file_count['left'] += 1
                    elif int(row['label_id_corrected']) == 2:
                        # No else because of skipped class
                        total_count['right'] += 1
                        in_file_count['right'] += 1
            
            file_count += 1
            if in_file_count['front'] >= int(row_count * (threshold)):
                front_csv.append(csv_path)
                # Tracks number of predicted classes to be removed
                for key in in_file_count.keys():
                    pred_removed[key] += in_file_count[key]
            
    return total_count, front_csv, file_count, pred_removed

def undersampling(label_folder, video_folder, front_csv):
    # Create sampled directories
    parent_label_dir = os.path.dirname(label_folder)
    sampled_labels = os.path.join(parent_label_dir, 'sampled_labels')
    os.makedirs(sampled_labels, exist_ok=True)
    print(f"Verified folder: {sampled_labels}")

    parent_vid_dir = os.path.dirname(video_folder)
    sampled_videos = os.path.join(parent_vid_dir, 'sampled_videos')
    os.makedirs(sampled_videos, exist_ok=True)
    print(f"Verified folder: {sampled_videos}")

    # Copies csv files with heterogenous labels along with their corresponding videos to sampled folders.
    for csv_file in os.listdir(label_folder):
        if csv_file.endswith(".csv"):
            
            csv_path = os.path.join(label_folder, csv_file)
            if csv_path not in front_csv:
                shutil.copy(csv_path, sampled_labels)
                # Copy corresponding video file
                video_file = csv_file.replace("_labels.csv", ".mp4")
                video_path = os.path.join(video_folder, video_file)
                if os.path.exists(video_path):
                    shutil.copy(video_path, sampled_videos)
                print(f"Added to the sample: {csv_file} & {video_file}")
                
    return sampled_labels

def ratio(count):
    max_turn = max(count['left'], count['right'])
    if max_turn == 0:
        max_turn = 1  # To avoid division by zero
    print(f"Ratio (Front:Max Turn): {int(count['front']/max_turn)}:{int(max_turn/max_turn)}")

def main():
    # Input processed label folder path
    # processed_labels folder
    label_folder = r''
    # Input processed video folder path
    # segmented videos
    video_folder = r''  

    # Change percentage threshold as needed (ex. 0.99 for 99%)
    threshold = 1

    start = time.time()
    count, front_csv, file_count, pred_removed = class_counter(label_folder, threshold)

    # Display class instances
    for key in count.keys():
        print(f"{key.capitalize()} class: {count[key]}")
    ratio(count)

    print("-" * 30)
    print(f"Number of {threshold*100}% front class csv files: {len(front_csv)} / {file_count} csv files")
    
    print(f"Predicted number of classes to be removed: ")
    for key in pred_removed.keys():
        print(f"{key.capitalize()} class: {pred_removed[key]}")
    print("-" * 30)

    # Remove all front csv files if user agrees
    user_input = input("Remove all front csv and corresponding videos (Y/N)? ")
    if(user_input.lower() == 'y'):
        sampled_labels = undersampling(label_folder, video_folder, front_csv)

        count, front_csv, file_count, pred_removed = class_counter(sampled_labels, threshold)
        
        print("-" * 30)
        print("Post-undersampling class counts:")
        for key in count.keys():
            print(f"{key.capitalize()} class: {count[key]}")
        ratio(count)
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()