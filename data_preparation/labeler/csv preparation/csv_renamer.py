import os
import shutil
import time

def main():
    input_path = r''
    path_dir = os.path.dirname(input_path)
    
    renamed_dir = os.path.join(path_dir, 'renamed_csv') 
    os.makedirs(renamed_dir, exist_ok=True)
    print(f"Verified folder: {renamed_dir}")
    print("Renamed files will be saved in the above directory.") 

    start = time.time()
    for filename in os.listdir(input_path):
        if filename.endswith(".csv"):
            new_filename = filename.replace("_labels", "")
            shutil.copy(os.path.join(input_path, filename), os.path.join(renamed_dir, new_filename)) 
            print(f"Renamed: {filename} to {new_filename}")
    end = time.time()

    print(f"Time taken to rename: {end - start:.2f} seconds.")

if __name__ == "__main__":
    main()