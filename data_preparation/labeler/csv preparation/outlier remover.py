import csv
import os
import shutil
import time

def fixing_outlier(labels_dir):
    # Locating and reading CSV files
    yaw_limit = 1.5 * 1.5 # Equal to the squared of turn_threshold 

    data_dir = os.path.dirname(labels_dir)
    processed_lbl_folder = os.path.join(data_dir, 'processed_labels') 
    os.makedirs(processed_lbl_folder, exist_ok=True)
    print(f"Verified folder: {processed_lbl_folder}")
    
    total_outliers = 0

    for csv_file in os.listdir(labels_dir):
        if csv_file.endswith("_labels.csv"):
            csv_path = os.path.join(labels_dir, csv_file)
            if os.path.exists(os.path.join(processed_lbl_folder, csv_file)):
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

                # Initialize corrected label id according to corrected yaw degrees
                # change append below
                turn_threshold = 1.5
                if float(label_yaw[i]) > turn_threshold:
                    label_id_corrected.append('2')
                elif float(label_yaw[i]) < -turn_threshold:
                    label_id_corrected.append('1')
                else:
                    label_id_corrected.append('0')

                # An outlier is detected if it is dissimilar from both neighbors
                if int(label_id_corrected[i-1]) == int(label_ids[i+1]) and int(ids) != int(label_id_corrected[i-1]):
                    # Keep lone outlier with high yaw_degrees
                    if float(label_yaw[i]) > yaw_limit:
                        # A high yaw degree is retained and influences the next neighbor 
                        # So the user has enough time for a direction
                        label_yaw[i+1] = float(label_yaw[i+1]) + (float(label_yaw[i]) - yaw_limit)
                        label_yaw_corrected.append(float(label_yaw[i]) - ((float(label_yaw[i]) - yaw_limit)))
                    elif float(label_yaw[i]) < -(yaw_limit):
                        label_yaw[i+1] = float(label_yaw[i+1]) + (float(label_yaw[i]) + 2.25)
                        label_yaw_corrected.append(float(label_yaw[i]) - ((float(label_yaw[i]) + yaw_limit)))
                    else: 
                        label_id_corrected[i] = label_ids[i-1] # Correct the outlier  
                        label_yaw_corrected.append(float(label_yaw[i]))
                        counter += 1
                else:
                    label_yaw_corrected.append(float(label_yaw[i]))

            print(f"{os.path.splitext(csv_file)[0]}.csv -> No. of outliers: {counter}")
            total_outliers += counter

            # Providing the corresponding corrected label names
            for label_id in label_id_corrected:
                if label_id == '0':
                    label_name_corrected.append('FRONT')
                elif label_id == '1':
                    label_name_corrected.append('LEFT')
                elif label_id == '2':
                    label_name_corrected.append('RIGHT')
                else:
                    label_name_corrected.append('SKIPPED')

            # All processed csv files are placed to a new folder 
            processed_csv_path = os.path.join(processed_lbl_folder, csv_file)
            shutil.copy(csv_path, processed_csv_path) 
   
            # Read the file again from the beginning
            with open(processed_csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Now write the corrected data
            with open(processed_csv_path, 'w', newline='') as f:
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
                            
    print(f"\nTotal outliers corrected across all files: {total_outliers}")
    return

def main():
    # labels folder
    labels_dir = r''
    start = time.time()
    print("-" * 30)
    print("Starting CSV Outlier Fixing\n")
    
    fixing_outlier(labels_dir)
    end = time.time()
    
    print(f"Processing time: {round(end-start, 2)} seconds or {round((end-start)/60, 2)} minutes\n")

if __name__ == "__main__":
    main()