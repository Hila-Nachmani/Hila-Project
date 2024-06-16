# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 20:41:57 2024

@author: hilan
"""
import os
import csv 
import zipfile
import shutil
import tensorflow as tf

def find_path(file_name):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, file_name)
    print("File path:")
    print(file_path)
    return file_path

def get_subfolders_and_zipfiles(folder_path):
    # Get the list of subfolders and zip files in the specified folder path
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    zipfiles = [f for f in os.listdir(folder_path) if f.endswith('.zip')]
    print("Subfolders in the folder:")
    if subfolders:
        for subfolder in subfolders:
            print(subfolder)
    else:
        print("None")
    
    print("\nZip files in the folder:")
    if zipfiles:
        for zipfile in zipfiles:
            print(zipfile)
    else:
        print("None")

    return subfolders, zipfiles
   
def unzip_files(folder_path, zipfiles):
    # Unzip each zip file in the specified folder
    
    val_data_folder = os.path.join(folder_path, "nsynth-valid")
    os.makedirs(val_data_folder, exist_ok=True)
    
    for zipfile_name in zipfiles:
        zipfile_path = os.path.join(folder_path, zipfile_name)
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        print(f"Unzipped: {zipfile_name}")
        
def list_files_in_folder(folder_path):
    """_summary_

    Args:
        folder_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory path.")
        return []

    # Get a list of all files in the directory
    files = os.listdir(folder_path)

    # Create a list to store the full paths of files
    file_paths = [os.path.join(folder_path, file_name) for file_name in files]

    # Return the list of file names and their corresponding full paths
    return files, file_paths

def move_problematic_file(file_path, dest_folder):
    """
    Move a problematic file to a destination folder.

    Args:
        file_path (string): Path to the problematic file.
        dest_folder (string): Path to the destination folder.
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    try:
        # Move the file to the destination folder
        shutil.move(file_path, dest_folder)
        print(f"Successfully moved '{file_path}' to '{dest_folder}'.")
    except Exception as e:
        print(f"Error moving '{file_path}': {e}")

def create_directory_if_not_exists(directory):
    """
    Create a directory if it does not exist.

    Args:
        directory (str): Directory path to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_notes_to_file(file_paths, notes, output_file):
    """
    Save a list of detected notes to a CSV file.

    Args:
        file_paths (list): List of audio file paths.
        notes (list): List of detected notes.
        output_file (str): The CSV file to save the notes to.
    """
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['file_path', 'closest_note'])
        for path, note in zip(file_paths, notes):
            writer.writerow([path, note])
'''         
def read_detected_notes(file_path):
    detected_notes = {}
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                file_name = row[0].strip()
                detected_note = row[1].strip()
                detected_notes[file_name] = detected_note
    except Exception as e:
        print(f"Error reading detected notes from CSV: {e}")
    
    return detected_notes
'''
        
'''
def create_dataset(data_dir, batch_size=32):
    # Function to load and preprocess the dataset
    def preprocess(file_path):
        # Load the audio file
        audio = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
        # Extract features (e.g., spectrogram)
        spectrogram = tf.signal.stft(tf.squeeze(audio.audio, axis=-1), frame_length=256, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        # Normalize the spectrogram
        spectrogram = tf.math.log(spectrogram + 1e-6)
        # Add a channel dimension
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
        # Pad to ensure consistent shape
        spectrogram = tf.image.resize_with_pad(spectrogram, 128, 126)
        return spectrogram, 0  # Dummy label, replace with actual label if available

    # Create a dataset of file paths
    list_ds = tf.data.Dataset.list_files(data_dir + '/*.wav')
    # Load and preprocess the audio files
    dataset = list_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
'''

    
    
'''
def find_path(file_name):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, file_name)
    print("File path:")
    print(file_path)
    return file_path

def get_label_from_file_path(file_path):
    # Split the file path using the directory separator
    parts = file_path.split(os.sep)
    # Extract the file name from the path
    file_name = parts[-1]
    # Extract the numeric part of the file name
    label_str = ''.join(filter(str.isdigit, file_name))
    # Convert the extracted numeric part to an integer
    label_int = int(label_str)
    return label_int

def adjust_label(label, min_label, max_label, num_piano_notes):
    # Calculate the range of the original labels
    label_range = max_label - min_label + 1
    # Calculate the scaling factor
    scale_factor = num_piano_notes / label_range
    # Scale the label to fit within the range of piano notes
    adjusted_label = int((label - min_label) * scale_factor)
    return adjusted_label

def get_min_max_labels(file_paths):
    # Initialize min and max labels with the first file path
    min_label = max_label = get_label_from_file_path(file_paths[0])

    # Iterate through all file paths to find the min and max labels
    for file_path in file_paths[1:]:
        label = get_label_from_file_path(file_path)
        min_label = min(min_label, label)
        max_label = max(max_label, label)

    return min_label, max_label

def map_numeric_to_label(numeric_part):
    # Convert the numeric part to an integer
    numeric_value = int(numeric_part)
    # Map the numeric value to the corresponding piano note label
    # Here, you can implement your logic to map the numeric value to the label
    # This could involve scaling, shifting, or any other transformation based on your dataset
    label = numeric_value - min_numeric_value  # Adjust according to your dataset
    return label


def rename_files_to_numbers(directory):
    """
    Rename files in a directory to sequential numbers.
    
    Args:
    - directory (str): Path to the directory containing the files.
    """
    files = os.listdir(directory)
    files.sort()  # Sort files alphabetically
    
    for i, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        new_filename = f"{i + 1}.wav"  # Renamed filename with sequential number and extension
        new_path = os.path.join(directory, new_filename)
        
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_filename}")

def copy_files_to_validation(train_data_folder, val_data_folder, validation_ratio):
    """
    Randomly copy files from the training dataset to the validation dataset.

    Parameters:
    - train_data_folder (str): Path to the folder containing training data.
    - val_data_folder (str): Path to the folder where validation data will be copied.
    - validation_ratio (float): Ratio of files to copy to the validation dataset (between 0 and 1).

    Returns:
    - None
    """
    # Ensure that the validation ratio is between 0 and 1
    if not 0 <= validation_ratio <= 1:
        raise ValueError("Validation ratio must be between 0 and 1.")

    # Create the validation data folder if it doesn't exist
    if not os.path.exists(val_data_folder):
        os.makedirs(val_data_folder)

    # Get the list of files in the training data folder
    files = os.listdir(train_data_folder)

    # Calculate the number of files to copy to validation
    num_files_to_copy = int(len(files) * validation_ratio)

    # Randomly select files to copy
    files_to_copy = random.sample(files, num_files_to_copy)

    # Copy selected files to the validation data folder
    for file_name in files_to_copy:
        src = os.path.join(train_data_folder, file_name)
        dst = os.path.join(val_data_folder, file_name)
        print(src,dst)
        shutil.copy(src, dst)

    print(f"{num_files_to_copy} files copied to validation dataset.")

"""
extract_dir = "Notes-Dataset"
# Create a ZipFile object
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    # Extract all contents into the specified directory
    zip_ref.extractall(extract_dir)
print("Files extracted successfully to:", extract_dir)
"""

def get_subfolders_and_zipfiles(folder_path):
    # Get the list of subfolders and zip files in the specified folder path
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    zipfiles = [f for f in os.listdir(folder_path) if f.endswith('.zip')]
    print("Subfolders in the folder:")
    if subfolders:
        for subfolder in subfolders:
            print(subfolder)
    else:
        print("None")
    
    print("\nZip files in the folder:")
    if zipfiles:
        for zipfile in zipfiles:
            print(zipfile)
    else:
        print("None")

    return subfolders, zipfiles

# Get and print the list of subfolders and zip files


def extract_zip(zip_file_path, extract_to):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Successfully extracted '{zip_file_path}' to '{extract_to}'")


def unzip_files(folder_path, zipfiles):
    # Unzip each zip file in the specified folder
    
    val_data_folder = os.path.join(folder_path, "nsynth-valid")
    os.makedirs(val_data_folder, exist_ok=True)
    
    for zipfile_name in zipfiles:
        zipfile_path = os.path.join(folder_path, zipfile_name)
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        print(f"Unzipped: {zipfile_name}")

def list_files_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory path.")
        return []

    # Get a list of all files in the directory
    files = os.listdir(folder_path)

    # Create a list to store the full paths of files
    file_paths = [os.path.join(folder_path, file_name) for file_name in files]
    
    # Generate full paths for each file and add them to the list
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        file_paths.append(file_path)

    # Return the list of file paths
    return files,file_paths
'''
