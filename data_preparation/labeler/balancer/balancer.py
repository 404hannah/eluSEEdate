import os
import csv
import shutil

def class_counter(label_folder):
    total_count = {'front': 0, 'left': 0, 'slight left': 0, 'slight right': 0, 'right': 0}
    front_csv = []
    file_count = 0
    pred_removed = 0

    # Counts the occurrences of each class label in the given folder.
    print(f"Counting instances of each class from the csv files.")

    for file in os.listdir(label_folder):
        if file.endswith(".csv"):
            csv_path = os.path.join(label_folder, file)
            in_file_count = {'front': 0}
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
                    elif int(row['label_id_corrected']) == 2:
                        total_count['slight left'] += 1
                    elif int(row['label_id_corrected']) == 3:
                        total_count['slight right'] += 1
                    elif int(row['label_id_corrected']) == 4: 
                        # No else because of skipped class
                        total_count['right'] += 1
            
            file_count += 1
            # Change percentage threshold as needed
            if in_file_count['front'] >= int(row_count * (1)):
                front_csv.append(csv_path)
                pred_removed += in_file_count['front']
            
    return total_count, front_csv, file_count, pred_removed

def undersampling(label_folder, front_csv):
    sampled_labels = os.path.join(label_folder, 'sampled_labels')
    os.makedirs(sampled_labels, exist_ok=True)
    print(f"Verified folder: {sampled_labels}")

    # Copies csv files with heterogenous labels to a folder.
    for file in os.listdir(label_folder):
        if file.endswith(".csv"):
            csv_path = os.path.join(label_folder, file)
            if csv_path not in front_csv:
                shutil.copy(csv_path, sampled_labels)
                print(f"Added to the sample: {file}")
    return sampled_labels

def main():
    # Input processed label folder path
    label_folder = r''
    count, front_csv, file_count, pred_removed = class_counter(label_folder)

    # Display class instances
    for key in count.keys():
        print(f"{key.capitalize()} class: {count[key]}")

    print("-" * 30)
    print(f"Number of all front class csv files: {len(front_csv)} / {file_count} csv files")
    print(f"Removing all front csv removes front instances: {pred_removed}")
    print("-" * 30)
    
    
    # Remove all front csv files if user agrees
    user_input = input("Remove all front csv (Y/N)? ")
    if(user_input.lower() == 'y'):
        sampled_labels = undersampling(label_folder, front_csv)

        count, front_csv, file_count, pred_removed = class_counter(sampled_labels)
        
        print("-" * 30)
        print("Post-undersampling class counts:")
        for key in count.keys():
            print(f"{key.capitalize()} class: {count[key]}")

if __name__ == "__main__":
    main()