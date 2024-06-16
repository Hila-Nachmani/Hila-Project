# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:15:48 2024

@author: hilan
"""
import numpy as np
from scipy.io import wavfile
import scipy.io.wavfile as wav
from models import *
from data_org import *
import matplotlib.pyplot as plt
import matplotlib 
import librosa
import librosa.display
from skimage.transform import resize
matplotlib.use("Agg")

def note_detect(file_path):
    try:
        """Detects the name of a note from an audio file and returns relevant plots.

        Args:
            file_path (string): Path to the audio file.

        Returns:
            closest_note (string): The name of the detected note.
            sound_plot (matplotlib.figure.Figure): Plot of the sound waveform.
            sound_plot_scaled (matplotlib.figure.Figure): Plot of the scaled sound waveform.
            fourier_plot (matplotlib.figure.Figure): Plot of the Fourier transform.
            peak_plot (matplotlib.figure.Figure): Plot of the peak detection.
        """
        # Define the frequency range for piano notes
        min_frequency = 27.5  # A0
        max_frequency = 4186  # C8

        # Generate frequencies array within the piano note range
        num_notes = 88  # Number of keys on a standard piano
        frequencies = np.logspace(np.log10(min_frequency), np.log10(max_frequency), num=num_notes)

        # Define the corresponding note names for piano keys
        note_names = ["A0", "A#0", "B0", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1",
                    "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2",
                    "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3",
                    "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4",
                    "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5",
                    "A5", "A#5", "B5", "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6",
                    "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7",
                    "A7", "A#7", "B7", "C8"]
        
        # Read the audio file
        sample_rate, audio_data = wavfile.read(file_path)
        
        file_length = len(audio_data) 
        f_s = sample_rate  # Sampling frequency



        # Plot sound waveform
        plt.figure(figsize=(10, 4))
        plt.plot(audio_data)
        plt.title('Sound Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        sound_plot = plt.gcf()
        plt.close('all')

        # Scale sound data to 0-1
        sound_scaled = audio_data / (2**15)

        # Plot scaled sound waveform
        plt.figure(figsize=(10, 4))
        plt.plot(sound_scaled)
        plt.title('Scaled Sound Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        sound_plot_scaled = plt.gcf()
        plt.close('all')

        # Perform Fourier transformation
        fourier = np.fft.fft(sound_scaled)
        fourier = np.absolute(fourier)

        # Plot Fourier spectrum
        plt.figure(figsize=(10, 4))
        plt.plot(fourier)
        plt.title('Fourier Transform')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        fourier_plot = plt.gcf()
        plt.close('all')

        # Peak detection
        imax = np.argmax(fourier[:file_length // 2])
        if imax == 0 or np.all(fourier == 0):
        # Handle the empty sequence case
            return None, None, None
        threshold = 0.3 * fourier[imax]
        i_begin = np.argmax(fourier[:imax] >= threshold)
        i_end = np.argmax(fourier[imax:] < threshold)
        peak_range = range(i_begin, i_end + imax)

        # Plot the peak detection
        plt.figure()
        plt.plot(peak_range, fourier[peak_range])
        plt.title("Peak Detection")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        peak_plot = plt.gcf()
        plt.close('all')

        # Calculate frequency
        freq = (imax * f_s) / file_length

        # Find the closest note
        closest_note_index = np.argmin(np.abs(freq - frequencies))
        closest_note = note_names[closest_note_index]

        return closest_note,fourier_plot, audio_data
    except np.core._exceptions._ArrayMemoryError as e:
        print(f"Error occurred during note detection: {e}")
        return None, None, None


def plot_audio_subset(audio_data, subset_size):
    # Use the audio data directly
    # Calculate the total number of samples in the audio data
    total_samples = len(audio_data)

    # Calculate the step size to create the subset
    step_size = max(total_samples // subset_size, 1)

    # Create a subset of the audio data by selecting every step_size-th sample
    subset_audio_data = audio_data[::step_size]

    # Create the time axis for the subset
    time_axis = np.arange(0, len(subset_audio_data)) * step_size

    # Plot the subset of the audio data
    plt.plot(time_axis, subset_audio_data)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Subset of Audio Data')
    plt.show()
    plt.close('all')

def convert_audio_to_spectrogram(audio_data, sample_rate=16000):
    """
    Convert audio data to a spectrogram using librosa.

    Args:
        audio_data (numpy.ndarray): Array containing audio data.
        sample_rate (int): Sampling rate of the audio (default: 16000).

    Returns:
        numpy.ndarray: Spectrogram of the audio data.
    """
    # Convert audio data to floating-point format (float32)
    audio_data_float32 = audio_data.astype(np.float32)
    # Compute mel spectrogram with sample rate 16000
    spectrogram = librosa.feature.melspectrogram(y=audio_data_float32, sr=sample_rate)
    return spectrogram

def normalize_spectrogram(spectrogram):
    # Normalize spectrogram data
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    normalized_spectrogram = (spectrogram - mean) / std
    return normalized_spectrogram

def resize_spectrogram(spectrogram, target_shape):
    # Resize spectrogram to target shape
    resized_spectrogram = resize(spectrogram, target_shape)
    return resized_spectrogram



















'''
def note_detect(file_path):
    # Define the frequency range for piano notes
    min_frequency = 27.5  # A0
    max_frequency = 4186  # C8

    # Generate frequencies array within the piano note range
    num_notes = 88  # Number of keys on a standard piano
    frequencies = np.logspace(np.log10(min_frequency), np.log10(max_frequency), num=num_notes)

    # Define the corresponding note names for piano keys
    note_names = ["A0", "A#0", "B0", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1",
                  "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2",
                  "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3",
                  "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4",
                  "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5",
                  "A5", "A#5", "B5", "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6",
                  "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7",
                  "A7", "A#7", "B7", "C8"]
	#storing the sound file as a numpy array
	#you can also use any other method to store the file as an np array 
    with open(file_path, 'rb') as file:
        # Read the audio file
        sample_rate, audio_data = wav.read(file)
    
    # Print sample rate and data type
    print("Sample Rate:", sample_rate)
    print("Data Type:", audio_data.dtype)
    
    # Print the audio data (if it's too large, it may not be practical to print)
    print("Audio Data:")
    print(audio_data)       
    file_length = len(audio_data) 
    f_s = sample_rate  #sampling frequency
    sound = audio_data #blank array
	
    plt.plot(sound)
    plt.show()

    num_channels = audio_data.shape[0] if audio_data.ndim > 1 else 1
    if num_channels == 1:
        print("Mono audio")
    elif num_channels == 2:
        print("Stereo audio")
    else:
        print("Unknown number of channels")    
    	
    sound=np.divide(sound,float(2**15)) #scaling it to 0 - 1
    counter = num_channels
	#-------------------------------------------
	
    plt.plot(sound)
    plt.show()

	#fourier transformation from numpy module
    fourier = np.fft.fft(sound)
    fourier = np.absolute(fourier)
    imax=np.argmax(fourier[0:int(file_length/2)]) #index of max element
		
    plt.plot(fourier)
    plt.show()

	#peak detection
    i_begin = -1
    threshold = 0.3 * fourier[imax]
    for i in range (0,imax+100):
        if fourier[i] >= threshold:
            if(i_begin==-1):
                i_begin = i				
            if(i_begin!=-1 and fourier[i]<threshold):
                break
    i_end = i
    imax = np.argmax(fourier[0:i_end+100])
	
    freq = (imax * f_s) / (file_length * counter)  # formula to convert index into sound frequency

    note = 0
    closest_note_index = np.argmin(np.abs(freq - frequencies))
    closest_note = note_names[closest_note_index]

    return closest_note   

     #first before changes

def load_and_preprocess_audio(audio_file_path, target_shape):
    """
    Load and preprocess audio data from file.

    Parameters:
    - file_path (str): Path to the audio file.
    - target_shape (tuple): Target shape to reshape the audio data.

    Returns:
    - audio_data (np.ndarray): Preprocessed audio data.
    """
    # Read the audio file
    sample_rate, audio_data = wav.read(audio_file_path)
    audio_data = audio_data.reshape(target_shape)
    
    # Ensure mono audio (if stereo, take the mean across channels)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if sample_rate != target_shape[0]:
        raise ValueError("Resampling not supported in this implementation. Please ensure audio is at the correct sample rate.")
    
    # Normalize the audio data
    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    
    # Pad or truncate the audio data to match the target shape
    if len(audio_data) < target_shape[1]:
        audio_data = np.pad(audio_data, (0, target_shape[1] - len(audio_data)))
    elif len(audio_data) > target_shape[1]:
        audio_data = audio_data[:target_shape[1]]
    
    # Reshape the audio data to match the target shape
    audio_data = audio_data.reshape(target_shape)
    
    return audio_data
'''
'''
#first method
def load_and_preprocess_audio(file_path, target_shape):
    # Load the audio file
    sample_rate, audio_data = wavfile.read(file_path)
    
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    
    # Normalize the audio data (if needed)
    #audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
    
    # Pad or truncate the audio data to the target length
    if len(audio_data) < target_shape[0]:
        # Pad the audio data with zeros
        padding = target_shape[0] - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), mode='constant')
    elif len(audio_data) > target_shape[0]:
        # Truncate the audio data
        audio_data = audio_data[:target_shape[0]]
    
    # Assuming you have a function to get labels from the file_path
    label = get_label_from_file_path(file_path)
    
    return audio_data.reshape(target_shape), label
    # Reshape the audio data to the desired shape
    audio_data = audio_data.reshape(-1, 1)
    
    return audio_data

def custom_data_generator(file_paths, batch_size, target_shape):
    while True:
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]
            batch_data = []
            batch_labels = []
            for file_path in batch_files:
                data, label = load_and_preprocess_audio(file_path, target_shape)
                batch_data.append(data)
                batch_labels.append(label)
                print("Label:", label)  # Debugging statement
            print("Batch Labels:", batch_labels)  # Debugging statement
            yield np.array(batch_data), np.array(batch_labels)



def train_model(audio_file_path, target_shape, num_classes):
    # Load and preprocess the audio data
    processed_audio_data = load_and_preprocess_audio(audio_file_path, target_shape)
    
    # Create the model using the processed audio data
    model = make_model(processed_audio_data, num_classes)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    train_data = ...  # Load training data
    val_data = ...    # Load validation data
    
    # Example code for training the model
    # Replace this with your actual training code
    model.fit(train_data, epochs=10, validation_data=val_data)
    
    # Once training is complete, you can save the model if needed
    # model.save('my_model.h5')
       
def plot_audio_subset(audio_data, subset_size):
    # Calculate the total number of samples in the audio data
    total_samples = len(audio_data)

    # Calculate the step size to create the subset
    step_size = max(total_samples // subset_size, 1)

    # Create a subset of the audio data by selecting every step_size-th sample
    subset_audio_data = audio_data[::step_size]

    # Create the time axis for the subset
    time_axis = np.arange(0, len(subset_audio_data)) * step_size

    # Plot the subset of the audio data
    plt.plot(time_axis, subset_audio_data)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Subset of Audio Data')
    plt.show()



import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import tensorflow as tf
from data_org import Find_Path, get_subfolders_and_zipfiles, unzip_files, list_files_in_folder

    
def note_detect(file_path):
    # Define the frequency range for piano notes
    min_frequency = 27.5  # A0
    max_frequency = 4186  # C8

    # Generate frequencies array within the piano note range
    num_notes = 88  # Number of keys on a standard piano
    frequencies = np.logspace(np.log10(min_frequency), np.log10(max_frequency), num=num_notes)

    # Define the corresponding note names for piano keys
    note_names = ["A0", "A#0", "B0", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1",
                  "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2",
                  "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3",
                  "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4",
                  "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5",
                  "A5", "A#5", "B5", "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6",
                  "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7",
                  "A7", "A#7", "B7", "C8"]
	#storing the sound file as a numpy array
	#you can also use any other method to store the file as an np array 
    with open(file_path, 'rb') as file:
        # Read the audio file
        sample_rate, audio_data = wav.read(file)
    
    # Print sample rate and data type
    print("Sample Rate:", sample_rate)
    print("Data Type:", audio_data.dtype)
    
    # Print the audio data (if it's too large, it may not be practical to print)
    print("Audio Data:")
    print(audio_data)       
    file_length = len(audio_data) 
    f_s = sample_rate  #sampling frequency
    sound = audio_data #blank array
	
    plt.plot(sound)
    #plt.show()

    num_channels = audio_data.shape[0] if audio_data.ndim > 1 else 1
    if num_channels == 1:
        print("Mono audio")
    elif num_channels == 2:
        print("Stereo audio")
    else:
        print("Unknown number of channels")    
    	
    sound=np.divide(sound,float(2**15)) #scaling it to 0 - 1
    counter = num_channels
	#-------------------------------------------
	
    plt.plot(sound)
    #plt.show()

	#fourier transformation from numpy module
    fourier = np.fft.fft(sound)
    fourier = np.absolute(fourier)
    imax=np.argmax(fourier[0:int(file_length/2)]) #index of max element
		
    plt.plot(fourier)
    #plt.show()

	#peak detection
    i_begin = -1
    threshold = 0.3 * fourier[imax]
    for i in range (0,imax+100):
        if fourier[i] >= threshold:
            if(i_begin==-1):
                i_begin = i				
            if(i_begin!=-1 and fourier[i]<threshold):
                break
    i_end = i
    imax = np.argmax(fourier[0:i_end+100])
	
    freq = (imax * f_s) / (file_length * counter)  # formula to convert index into sound frequency

    note = 0
    closest_note_index = np.argmin(np.abs(freq - frequencies))
    closest_note = note_names[closest_note_index]

    return closest_note    
# Locate the dataset
dataset_path = Find_Path("Notes-Dataset")

# Get subfolders and zip files in the dataset
subfolders, zipfiles = get_subfolders_and_zipfiles(dataset_path)

# Extract zipped files
if zipfiles:
    unzip_files(dataset_path, zipfiles)

# List all files in each subfolder
for subfolder_name in subfolders:
    subfolder_path = os.path.join(dataset_path, subfolder_name)
    files, file_paths = list_files_in_folder(subfolder_path)
    
    for file_path in file_paths:
        # Load audio data from file
        sample_rate, audio_data = wav.read(file_path)
        
        # Perform note detection (placeholder function)
        detected_note = note_detect(audio_data)
        
        # Print detected note
        print("Detected Note:", detected_note)


def train_and_validate_model(model, train_data, val_data, num_epochs, batch_size):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(num_epochs):
        # Train the model
        for batch_data, batch_labels in train_data:
            with tf.GradientTape() as tape:
                train_predictions = model(batch_data)
                train_loss = loss_fn(batch_labels, train_predictions)

            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Evaluate the model on training data
        train_loss, train_accuracy = evaluate_model(model, train_data, loss_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model on validation data
        val_loss, val_accuracy = evaluate_model(model, val_data, loss_fn)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Train Acc: {train_accuracy}, Val Loss: {val_loss}, Val Acc: {val_accuracy}")

    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, "Loss", "Training and Validation Loss")
    plot_metrics(train_accuracies, val_accuracies, "Accuracy", "Training and Validation Accuracy")

def evaluate_model(model, data, loss_fn):
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    losses = []

    for batch_data, batch_labels in data:
        predictions = model(batch_data)
        loss = loss_fn(batch_labels, predictions)
        accuracy_metric.update_state(batch_labels, predictions)
        losses.append(loss.numpy())

    mean_loss = np.mean(losses)
    accuracy = accuracy_metric.result().numpy()
    accuracy_metric.reset_states()

    return mean_loss, accuracy

def plot_metrics(train_metrics, val_metrics, ylabel, title):
    epochs = np.arange(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, label='Train')
    plt.plot(epochs, val_metrics, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

'''
# train_and_validate_model(model, train_data, val_data, num_epochs, batch_size)
