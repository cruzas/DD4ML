import os
import sys

def rename_files(directory, old_str, new_str):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if old_str in filename:
            # Construct new filename
            new_filename = filename.replace(old_str, new_str)
            # Rename the file
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    
    print("Files renamed successfully!")

if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: python rename.py directory old_str new_str")
        sys.exit(1)
    
    # Get arguments
    directory = sys.argv[1]
    old_str = sys.argv[2]
    new_str = sys.argv[3]

    # Rename files
    rename_files(directory, old_str, new_str)
