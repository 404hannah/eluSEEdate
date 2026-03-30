import os
import shutil
import time

def main():
    # Enter former and new initials
    former_initial = ''
    new_initial = ''

    directory = r''
    new_directory = os.path.join(directory, 'renamed_files')
    os.makedirs(new_directory, exist_ok=True)
    print(f"Verified folder: {new_directory}") 
    print("Renamed files will be saved in the above directory.") 

    start = time.time()
    for filename in os.listdir(directory):
        if filename.startswith(former_initial):
            new_filename = new_initial + filename[1:]
            shutil.copy(os.path.join(directory, filename), os.path.join(new_directory, new_filename)) 
            print(f"Renamed: {filename} to {new_filename}")
    end = time.time()

    print(f"Time taken to rename: {end - start:.2f} seconds.")

if __name__ == "__main__":
    main()