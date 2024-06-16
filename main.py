# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:15:48 2024

@author: hilan
"""

import os
TF_ENABLE_ONEDNN_OPTS=0
from data_org import *
from models import *
from audio_processing import *
#from user import *
plt.ion()


def main():
    
    dataset_path = find_path("Notes-Dataset")
    project_path = os.getcwd()
    bad_audio_path = find_path("not_good")
    subfolders, zipfiles = get_subfolders_and_zipfiles(dataset_path)
    
    train_data = {}
    train_resized_spectrograms=[]
    train_closest_notes = []
    
    val_data = {}
    val_resized_spectrograms=[]
    val_closest_notes = []
    test_data = {}
    
    test_resized_spectrograms=[]
    test_closest_notes = []
    
    output_train_file = 'detected_train_notes.csv'
    output_val_file = 'detected_val_notes.csv'
    output_test_file = 'detected_test_notes.csv'
    
    # Ensure output directories exist
    train_output_dir = os.path.join(project_path , 'detected_notes', 'train')
    val_output_dir = os.path.join(project_path , 'detected_notes', 'val')
    test_output_dir = os.path.join(project_path , 'detected_notes', 'test')
    
    create_directory_if_not_exists(train_output_dir)
    create_directory_if_not_exists(val_output_dir)
    create_directory_if_not_exists(test_output_dir)
    
    output_train_file = os.path.join(train_output_dir, 'detected_train_notes.csv')
    output_val_file = os.path.join(val_output_dir, 'detected_val_notes.csv')
    output_test_file = os.path.join(test_output_dir, 'detected_test_notes.csv')
    
    if zipfiles:
        unzip_files(dataset_path, zipfiles)
    subfolders, zipfiles = get_subfolders_and_zipfiles(dataset_path)

    train_path, valid_path, test_path = None, None, None

    for subfolder_name in subfolders:
        print(subfolder_name)
        subfolder_path = os.path.join(dataset_path, subfolder_name)
        print(subfolder_name, '-path:', subfolder_path)
        if subfolder_name == 'nsynth-train':  # Update condition to match folder name
            train_path = subfolder_path
        elif subfolder_name == 'nsynth-valid':  # Update condition to match folder name
            valid_path = subfolder_path
        elif subfolder_name == 'nsynth-test':  # Update condition to match folder name
            test_path = subfolder_path

        if train_path:
            train_files, train_file_paths = list_files_in_folder(train_path)
        else:
            train_files, train_file_paths = [], []

        if valid_path:
            val_files, val_file_paths = list_files_in_folder(valid_path)
        else:
            val_files, val_file_paths = [], []

        if test_path:
            test_files, test_file_paths = list_files_in_folder(test_path)
        else:
            test_files, test_file_paths = [], []

        # Print statements to indicate which paths were found
        if train_path:
            print(f"Train path found: {train_path}")
        if valid_path:
            print(f"Validation path found: {valid_path}")
        if test_path:
            print(f"Test path found: {test_path}")

        # Handle cases where none of the paths were found
        if not train_path and not valid_path and not test_path:
            print("No data paths found. Ensure your dataset structure is correct.")
    
    batch_size = 20
    total_train_files = len(train_file_paths)
    processed_train_files = 0
    reached_validation_phase= False
    target_shape = (128, 126)
        
    for i in range(0, len(train_file_paths), batch_size):
        batch_files = train_file_paths[i:i + batch_size]
        for file_path in batch_files:
            closest_note, fourier_plot, audio_data = note_detect(file_path)
            if fourier_plot is not None: 
                if len(train_closest_notes) < 30000:
                    train_closest_notes.append(closest_note)
                    spectrogram=convert_audio_to_spectrogram(audio_data, sample_rate=16000)
                    normalized_spectrogram = normalize_spectrogram(spectrogram)
                    resized_spectrogram = resize_spectrogram(normalized_spectrogram, target_shape)
                    train_resized_spectrograms.append(resized_spectrogram)
                    processed_train_files += 1
                    if len(train_closest_notes) > 18000 and not reached_validation_phase:
                        print("Moving to validation phase...")
                        reached_validation_phase = True
                        break
                else:
                    print("Train data array is full. Skipping...")  
                    continue
            else:
                print(f"Error processing '{file_path}': Fourier transform result is empty.")
                move_problematic_file(file_path, bad_audio_path)
                break
            
            if processed_train_files % 100  == 0:  # Print progress every 100 files
                print(f"Processed train files: {processed_train_files}/{total_train_files}")
        
        if reached_validation_phase:
            break
    if reached_validation_phase:
        pass   
    
    train_data = {
    'closest_notes': train_closest_notes,
    'resized_spectrograms':train_resized_spectrograms
    }
    print("done with train files")
    
    save_notes_to_file(train_file_paths, train_closest_notes, output_train_file)
    print(f"Saved detected train notes to '{output_train_file}'")

    
    total_val_files = len(val_file_paths)
    processed_val_files = 0
    for i in range(0, len(val_file_paths), batch_size):
        batch_files = val_file_paths[i:i + batch_size]
        for file_path in batch_files:
            closest_note, fourier_plot, audio_data = note_detect(file_path)
            if fourier_plot is not None: 
                val_closest_notes.append(closest_note)
                spectrogram=convert_audio_to_spectrogram(audio_data, sample_rate=16000)
                normalized_spectrogram = normalize_spectrogram(spectrogram)
                resized_spectrogram = resize_spectrogram(normalized_spectrogram, target_shape)
                val_resized_spectrograms.append(resized_spectrogram)
            else:
                print(f"Error processing '{file_path}': Fourier transform result is empty.")
                move_problematic_file(file_path, bad_audio_path)
            processed_val_files += 1
            if processed_val_files % 100 == 0:  # Print progress every 100 files
                print(f"Processed validation files: {processed_val_files}/{total_val_files}")
    val_data = {
    'closest_notes': val_closest_notes,
    'resized_spectrograms':val_resized_spectrograms
}
    print("done with validation files")
    
    save_notes_to_file(val_file_paths, val_closest_notes, output_val_file)
    print(f"Saved detected validation notes to '{output_val_file}'")

    
    total_test_files = len(test_file_paths)
    processed_test_files = 0
    for i in range(0, len(test_file_paths), batch_size):
        batch_files = test_file_paths[i:i + batch_size]
        for file_path in batch_files:
            closest_note, fourier_plot, audio_data = note_detect(file_path)
            if fourier_plot is not None: 
                test_closest_notes.append(closest_note)
                spectrogram=convert_audio_to_spectrogram(audio_data, sample_rate=16000)
                normalized_spectrogram = normalize_spectrogram(spectrogram)
                resized_spectrogram = resize_spectrogram(normalized_spectrogram, target_shape)
                test_resized_spectrograms.append(resized_spectrogram)
            else:
                print(f"Error processing '{file_path}': Fourier transform result is empty.")
                move_problematic_file(file_path, bad_audio_path)
            processed_test_files += 1
            if processed_test_files % 100 == 0:  # Print progress every 100 files
                print(f"Processed test files: {processed_test_files}/{total_test_files}")
    test_data = {
    'closest_notes': test_closest_notes,
    'resized_spectrograms':test_resized_spectrograms
}
    print("done with test files")
    
    save_notes_to_file(test_file_paths, test_closest_notes, output_test_file)
    print(f"Saved detected test notes to '{output_test_file}'")

    
    num_channels = 1
    input_shape = (train_data['resized_spectrograms'][0].shape[0], train_data['resized_spectrograms'][0].shape[1], num_channels)
    print(input_shape)
    num_classes = 88  # Ex ample number of classes
    # Build the CNN model
    model = build_model(input_shape, num_classes)
    
    sample_input = np.zeros((1,) + input_shape)  # Create a sample input tensor
    output_shape = model.layers[0](sample_input).shape
    print("Output shape after first Conv2D layer:", output_shape)
    # Print a summary of the model architecture 
    model.summary()

    #training and validation
    
    batch_size = 25
    epochs = 30
    model_path, history, encoder_path ,checkpoint_path = train_and_valid(model, train_data, val_data, batch_size, epochs)
    label_encoder = joblib.load(encoder_path)
    '''
    encoder_path = os.path.join(find_path("ML-Project"), 'saved_encoders', 'encoder_20240414003903.pkl')
    model_path = os.path.join(find_path("ML-Project"), 'saved_models', 'model_20240414003902.h5')
    checkpoint_path = os.path.join(find_path("ML-Project"), 'saved_encoders', 'encoder_20240414003903.pkl')
    '''
    label_encoder = joblib.load(encoder_path)
    
    test_model(test_data, model_path, label_encoder, print_predictions=True)
    
if __name__ == "__main__":
    main()

'''
def main():
    
    dataset_path = find_path("Notes-Dataset")
    bad_audio_path = find_path("not_good")
    subfolders, zipfiles = get_subfolders_and_zipfiles(dataset_path)
    train_data = {}
    #train_spectrograms=[]
    #train_normalized_spectrograms=[]
    train_resized_spectrograms=[]
    #train_sound_plots = []
    #train_sound_plots_scaled = []
    #train_fourier_plots = []
    #train_peak_plots = []
    #train_audio_data=[]
    train_closest_notes = []
    val_data = {}
    #val_spectrograms=[]
    #val_normalized_spectrograms=[]
    val_resized_spectrograms=[]
    #val_sound_plots = []
    #val_sound_plots_scaled = []
    #val_fourier_plots = []
    #val_peak_plots = []
    #val_audio_data=[]
    val_closest_notes = []
    test_data = {}
    #test_spectrograms=[]
    #test_normalized_spectrograms=[]
    test_resized_spectrograms=[]
    #test_sound_plots = []
    #test_sound_plots_scaled = []
    #test_fourier_plots = []
    #test_peak_plots = []
    #test_audio_data=[]
    test_closest_notes = []
    if zipfiles:
      unzip_files(dataset_path, zipfiles)
    subfolders, zipfiles = get_subfolders_and_zipfiles(dataset_path)
    for subfolder_name in subfolders:
        print(subfolder_name)
        subfolder_path = os.path.join(dataset_path, subfolder_name)
        print(subfolder_name,'-path:', subfolder_path)
        if subfolder_name=='nsynth-train':
            train_path=subfolder_path
        elif subfolder_name=='nsynth-valid':
            valid_path=subfolder_path
        elif subfolder_name=='nsynth-test':
            test_path=subfolder_path
         
        #files, file_paths = list_files_in_folder(subfolder_path)
    
    train_files, train_file_paths = list_files_in_folder(train_path)
    batch_size = 20
    total_train_files = len(train_file_paths)
    processed_train_files = 0
    reached_validation_phase= False
    target_shape = (128, 126)
    
    for i in range(0, len(train_file_paths), batch_size):
        batch_files = train_file_paths[i:i + batch_size]
        for file_path in batch_files:
            closest_note, sound_plot, sound_plot_scaled, fourier_plot, peak_plot, audio_data = note_detect(file_path)
            if fourier_plot is not None: 
                if len(train_closest_notes) < 30000:
                    #train_sound_plots_scaled.append(sound_plot_scaled)
                    #train_fourier_plots.append(fourier_plot)
                    #train_peak_plots.append(peak_plot)
                    #train_sound_plots.append(sound_plot)
                    train_closest_notes.append(closest_note)
                    #train_audio_data.append(audio_data)
                    spectrogram=convert_audio_to_spectrogram(audio_data, sample_rate=16000)
                    #train_spectrograms.append(spectrogram)
                    normalized_spectrogram = normalize_spectrogram(spectrogram)
                    #train_normalized_spectrograms.append(normalized_spectrogram)
                    resized_spectrogram = resize_spectrogram(normalized_spectrogram, target_shape)
                    train_resized_spectrograms.append(resized_spectrogram)
                    #plot_audio_subset(audio_data, subset_size=500)
                    processed_train_files += 1
                    if len(train_closest_notes) > 10000 and not reached_validation_phase:
                        print("Moving to validation phase...")
                        reached_validation_phase = True
                        break
                else:
                    print("Train data array is full. Skipping...")  
                    continue
            else:
                print(f"Error processing '{file_path}': Fourier transform result is empty.")
                move_problematic_file(file_path, bad_audio_path)
                break
            
            if processed_train_files % 100  == 0:  # Print progress every 100 files
                print(f"Processed train files: {processed_train_files}/{total_train_files}")
        
        if reached_validation_phase:
            break
    if reached_validation_phase:
        pass   
    
    train_data = {
    'closest_notes': train_closest_notes,
    #'sound_plots': train_sound_plots,
    #'sound_plots_scaled': train_sound_plots_scaled,
    #'fourier_plots': train_fourier_plots,
    #'peak_plots': train_peak_plots,
    #'audio_data': train_audio_data,
    #'spectrograms':train_spectrograms,
    #'normalized_spectrograms':train_normalized_spectrograms,
    'resized_spectrograms':train_resized_spectrograms
    }
    print("done with train files")
    
    val_files, val_file_paths = list_files_in_folder(valid_path)
    total_val_files = len(val_file_paths)
    processed_val_files = 0
    for i in range(0, len(val_file_paths), batch_size):
        batch_files = val_file_paths[i:i + batch_size]
        for file_path in batch_files:
            closest_note, sound_plot, sound_plot_scaled, fourier_plot, peak_plot, audio_data = note_detect(file_path)
            if fourier_plot is not None: 
                #val_sound_plots_scaled.append(sound_plot_scaled)
                #val_fourier_plots.append(fourier_plot)
                #val_peak_plots.append(peak_plot)
                val_closest_notes.append(closest_note)
                #val_sound_plots.append(sound_plot)
                #val_audio_data.append(audio_data)
                spectrogram=convert_audio_to_spectrogram(audio_data, sample_rate=16000)
                #val_spectrograms.append(spectrogram)
                normalized_spectrogram = normalize_spectrogram(spectrogram)
                #val_normalized_spectrograms.append(normalized_spectrogram)
                resized_spectrogram = resize_spectrogram(normalized_spectrogram, target_shape)
                val_resized_spectrograms.append(resized_spectrogram)
                #plot_audio_subset(audio_data, subset_size=500)
            else:
                print(f"Error processing '{file_path}': Fourier transform result is empty.")
                move_problematic_file(file_path, bad_audio_path)
            processed_val_files += 1
            if processed_val_files % 100 == 0:  # Print progress every 100 files
                print(f"Processed validation files: {processed_val_files}/{total_val_files}")
    val_data = {
    'closest_notes': val_closest_notes,
    #'sound_plots': val_sound_plots,
    #'sound_plots_scaled': val_sound_plots_scaled,
    #'fourier_plots': val_fourier_plots,
    #'peak_plots': val_peak_plots,
    #'audio_data':val_audio_data,
    #'spectrograms':val_spectrograms,
    #'normalized_spectrograms':val_normalized_spectrograms,
    'resized_spectrograms':val_resized_spectrograms
}
    print("done with validation files")
    
    test_files, test_file_paths = list_files_in_folder(test_path)
    total_test_files = len(test_file_paths)
    processed_test_files = 0
    for i in range(0, len(test_file_paths), batch_size):
        batch_files = test_file_paths[i:i + batch_size]
        for file_path in batch_files:
            closest_note, sound_plot, sound_plot_scaled, fourier_plot, peak_plot, audio_data = note_detect(file_path)
            if fourier_plot is not None: 
                #test_sound_plots_scaled.append(sound_plot_scaled)
                #test_fourier_plots.append(fourier_plot)
                #test_peak_plots.append(peak_plot)
                test_closest_notes.append(closest_note)
                #test_sound_plots.append(sound_plot)
                #test_audio_data.append(audio_data)
                spectrogram=convert_audio_to_spectrogram(audio_data, sample_rate=16000)
                #test_spectrograms.append(spectrogram)
                normalized_spectrogram = normalize_spectrogram(spectrogram)
                #test_normalized_spectrograms.append(normalized_spectrogram)
                resized_spectrogram = resize_spectrogram(normalized_spectrogram, target_shape)
                test_resized_spectrograms.append(resized_spectrogram)
                #plot_audio_subset(audio_data, subset_size=500)
                
            else:
                print(f"Error processing '{file_path}': Fourier transform result is empty.")
                move_problematic_file(file_path, bad_audio_path)
            processed_test_files += 1
            if processed_test_files % 100 == 0:  # Print progress every 100 files
                print(f"Processed test files: {processed_test_files}/{total_test_files}")
    test_data = {
    'closest_notes': test_closest_notes,
    #'sound_plots': test_sound_plots,
    #'sound_plots_scaled': test_sound_plots_scaled,
    #'fourier_plots': test_fourier_plots,
    #'peak_plots': test_peak_plots,
    #'audio_data':test_audio_data,
    #'spectrograms':test_spectrograms,
    #'normalized_spectrograms':test_normalized_spectrograms,
    'resized_spectrograms':test_resized_spectrograms
}
    print("done with test files")
    
    train_audio_data = np.array(train_data['audio_data'])
    train_spectrograms =convert_audio_to_spectrograms(train_audio_data, sample_rate=16000)
    # Normalize spectrograms
    train_normalized_spectrograms = normalize_spectrograms(train_spectrograms)
    print('finished normalizing spectograms')
    # Resize spectrograms
    target_shape = (128, 126)
    train_resized_spectrograms = resize_spectrograms(train_normalized_spectrograms, target_shape)
    print('finished resizing spectograms')
    print('train_spectograms shape:',train_resized_spectrograms.shape)
    
    
    num_channels = 1
    input_shape = (train_data['resized_spectrograms'][0].shape[0], train_data['resized_spectrograms'][0].shape[1], num_channels)
    print(input_shape)
    num_classes = 88  # Ex ample number of classes
    # Build the CNN model
    model = build_model(input_shape, num_classes)
    
    sample_input = np.zeros((1,) + input_shape)  # Create a sample input tensor
    output_shape = model.layers[0](sample_input).shape
    print("Output shape after first Conv2D layer:", output_shape)
    # Print a summary of the model architecture 
    model.summary()

    #training and validation
    
    batch_size = 25
    epochs = 30
    model_path, history, encoder_path ,checkpoint_path = train_and_valid(model, train_data, val_data, batch_size, epochs)
    label_encoder = joblib.load(encoder_path)
    
    encoder_path = os.path.join(find_path("ML-Project"), 'saved_encoders', 'encoder_20240414003903.pkl')
    model_path = os.path.join(find_path("ML-Project"), 'saved_models', 'model_20240414003902.h5')
    checkpoint_path = os.path.join(find_path("ML-Project"), 'saved_encoders', 'encoder_20240414003903.pkl')
    
    label_encoder = joblib.load(encoder_path)
    
    test_model(test_data, model_path, label_encoder, print_predictions=True)
    
if __name__ == "__main__":
    main()
''' 
    
    
    
    
    
'''
    # Define input shape and number of classes
    train_audio_data = np.array(train_data['audio_data'])  
    train_audio_spectrograms = convert_sound_plots_to_spectrograms(train_audio_data, sample_rate=16000)
    val_audio_data = np.array(val_data['audio_data']) 
    val_audio_spectrograms = convert_sound_plots_to_spectrograms(val_audio_data, sample_rate=16000)
    
    print (train_audio_spectrograms)
    print (val_audio_spectrograms)
    
    t_height, t_width = train_audio_spectrograms.shape[1:] 
    t_channels = 1 
    v_height, v_width = val_audio_spectrograms.shape[1:] 
    v_channels = 1 
   
    t_height, t_width, t_channels = train_audio_spectrograms.shape
    v_height, v_width, v_channels = val_audio_spectrograms.shape
    
    print("Train audio spectrograms shape:", train_audio_spectrograms.shape)
    print("Validation audio spectrograms shape:", val_audio_spectrograms.shape)
    
    # Reshape the input data to match the expected input shape of the model
    train_audio_spectrograms = train_audio_spectrograms.reshape(-1, t_height, t_width, t_channels)
    val_audio_spectrograms = val_audio_spectrograms.reshape(-1, v_height, v_width, v_channels)
    
    print("Reshaped Train audio spectrograms shape:", train_audio_spectrograms.shape)
    print("Reshaped Validation audio spectrograms shape:", val_audio_spectrograms.shape)
    
    input_shape = input_shape = train_audio_spectrograms.shape 
    
    
    # Assign the reshaped spectrograms to the train and validation data dictionaries
    train_data['spectrograms'] = train_audio_spectrograms
    val_data['spectrograms'] = val_audio_spectrograms 
    '''
    
    
'''
    # Define input shape and number of classes
    train_audio_data = np.array(train_data['audio_data'])  
    train_audio_spectrograms = convert_sound_plots_to_spectrograms(train_audio_data, sample_rate=16000)
    t_height, t_width, t_channels = train_audio_spectrograms.shape
    print("Height:", t_height)
    print("Width:", t_width)
    print("Channels:", t_channels)
    input_shape = (t_height, t_width, t_channels)
    num_classes = 88  # Example number of classes

    # Build the CNN model
    model = build_model(input_shape, num_classes)
    
    # Print a summary of the model architecture
    model.summary()
    '''


    








'''
NUM_PIANO_NOTES = 88  
def main():
    # Locate the dataset
    dataset_path = find_path("Notes-Dataset")

    # Get subfolders and zip files in the dataset
    subfolders, zipfiles = get_subfolders_and_zipfiles(dataset_path)

    # Extract zipped files
    #if zipfiles:
        #unzip_files(dataset_path, zipfiles)

    subfolders, zipfiles = get_subfolders_and_zipfiles(dataset_path)
    # List all files in each subfolderpy
    for subfolder_name in subfolders:
        print(subfolder_name)
        subfolder_path = os.path.join(dataset_path, subfolder_name)
        files, file_paths = list_files_in_folder(subfolder_path)
        
        for file_path in file_paths:
            #Perform note detection on each audio file
            detected_note = note_detect(file_path)
            # Print detected note
            print("Detected Note:", detected_note)
            sample_rate, audio_data = wav.read(file_path)
            plot_audio_subset(audio_data, subset_size=1000)
        print('done')
        
    train_data_folder = 'none'
    val_data_folder = 'none'
    for subfolder_name in subfolders:
        if(subfolder_name =='nsynth-train'):
            train_data_folder = os.path.join(dataset_path, subfolder_name)
        elif(subfolder_name =='nsynth-valid'):
            val_data_folder = os.path.join(dataset_path, subfolder_name)
    if train_data_folder is None or val_data_folder is None:
        print("Error: Train or validation data folder not found.")
        return
             
    train_files, train_file_paths = list_files_in_folder(train_data_folder)
    val_files, val_file_paths = list_files_in_folder(val_data_folder) 
    train_min_label, train_max_label = get_min_max_labels(train_file_paths)
    val_min_label, val_max_label = get_min_max_labels(val_file_paths)
    label_range = (min(train_min_label, val_min_label), max(train_max_label, val_max_label))

    validation_ratio = 0.2  # 20% of files will be copied to validation

    #copy_files_to_validation(train_data_folder, val_data_folder, validation_ratio)
    final_file_path='none'
    for file_path in file_paths:
        final_file_path=file_path
    num_epochs = 5
    batch_size = 100
    input_shape = (16000, 1, 1)
    sample_rate, audio_data = wav.read(final_file_path)
    audio_data = audio_data.reshape(-1, 1)  # Reshape audio data
    num_classes = 20000000
    model = make_model(input_shape, num_classes)  # Create the model
    
    target_shape = (16000, 1)
    audio_data = load_and_preprocess_audio(file_path, target_shape)
    
    train_generator = custom_data_generator(train_file_paths, batch_size, target_shape)
    val_generator = custom_data_generator(val_file_paths, batch_size, target_shape)
    
    train_labels_adjusted = [adjust_label(get_label_from_file_path(file_name), train_min_label, train_max_label, num_piano_notes) for file_name in train_files]
    val_labels_adjusted = [adjust_label(get_label_from_file_path(file_name), val_min_label, val_max_label, num_piano_notes) for file_name in val_files]
    
    # Use the get_label_from_file_path function to extract labels from file paths
    train_labels = [int(get_label_from_file_path(file_name)) for file_name in train_files]
    val_labels = [int(get_label_from_file_path(file_name)) for file_name in val_files]
    
    # Use LabelEncoder to convert string labels to integers
    label_encoder = LabelEncoder()
    all_labels = train_labels_adjusted + val_labels_adjusted
    label_encoder.fit(all_labels)
    
    train_labels_encoded = label_encoder.transform(train_labels_adjusted)
    val_labels_encoded = label_encoder.transform(val_labels_adjusted)
    
    print("Adjusted Train Labels:", train_labels_adjusted)
    print("Adjusted Validation Labels:", val_labels_adjusted)
    print("Encoded Train Labels:", train_labels_encoded)
    print("Encoded Validation Labels:", val_labels_encoded)
    
    all_labels = np.concatenate((train_labels_encoded, val_labels_encoded))
    encoded_labels = label_encoder.fit_transform(all_labels)
    print(encoded_labels)
    
    #rename_files_to_numbers(train_data_folder)
    label_encoder.fit(train_files)
    val_files_filtered = [file for file in val_files if file in train_files]
    val_labels_encoded = label_encoder.transform(val_files_filtered)
    train_labels_encoded = label_encoder.transform(train_files)
    
    print("Train Files:", train_files)
    print("Train Labels Encoded:", train_labels_encoded)
    
    # Print validation files and their encoded labels
    print("Validation Files:", val_files)
    print("Validation Labels Encoded:", val_labels_encoded)
    
    all_labels = np.concatenate((train_labels_encoded, val_labels_encoded))
    encoded_labels = label_encoder.fit_transform(all_labels)
    print(encoded_labels)
    
    # Transform filtered validation labels
    #val_labels_encoded = label_encoder.transform(val_files_filtered)


    target_length = 16000  # You can adjust this value as needed
    train_generator = custom_data_generator(train_file_paths, batch_size, target_shape)
    val_generator = custom_data_generator(val_file_paths, batch_size, target_shape)
    train_and_validate_model(model, train_generator, val_generator, num_epochs,batch_size)
    input_data = audio_data.reshape(-1, 32000, 1, 1)  # Reshape input data for the model
    predictions = model.predict(input_data)

if __name__ == "__main__":
    num_piano_notes = NUM_PIANO_NOTES
    main()

'''

'''
import os
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS=0
from data_org import *
from models import *
from audio_processing import *
from sklearn.preprocessing import LabelEncoder
plt.ion()


def main():
    # Locate the dataset
    dataset_path = find_path("Notes-Dataset")

    # Get subfolders and zip files in the dataset
    subfolders, zipfiles = get_subfolders_and_zipfiles(dataset_path)
    
    #Extract zipped files
    if zipfiles:
        unzip_files(dataset_path, zipfiles)
    
    val_data_folder = os.path.join(dataset_path, "nsynth-valid")
    os.makedirs(val_data_folder, exist_ok=True)
    
    print("Subfolders in the folder:", subfolders)
    
    train_data_folder = None
    val_data_folder = None
    
    for subfolder_name in subfolders:
        print("Subfolder:", subfolder_name)
        subfolder_path = os.path.join(dataset_path, subfolder_name)
        files, file_paths = list_files_in_folder(subfolder_path)
        if subfolder_name == 'nsynth-train':
            train_data_folder = subfolder_path
        elif subfolder_name == 'nsynth-valid':
            val_data_folder = subfolder_path
    print("Train Data Folder:", train_data_folder)
    print("Validation Data Folder:", val_data_folder)
    # Check for the existence of train and validation data folders
    if "nsynth-train" in subfolders:
        train_data_folder = os.path.join(dataset_path, "nsynth-train")
    if "nsynth-valid" in subfolders:
        val_data_folder = os.path.join(dataset_path, "nsynth-valid")
    if train_data_folder is None or val_data_folder is None:
        print("Error: Train or validation data folder not found.")
        return
    
    
    # List all files in each subfolder
    for subfolder_name in subfolders:
        print("Subfolder:", subfolder_name)
        subfolder_path = os.path.join(dataset_path, subfolder_name)
        rename_files_to_numbers(subfolder_path)
        files, file_paths = list_files_in_folder(subfolder_path)
        if subfolder_name == 'nsynth-train':
            train_data_folder = subfolder_path
        elif subfolder_name == 'nsynth-valid':
            val_data_folder = subfolder_path
    print("Train Data Folder:", train_data_folder)
    print("Validation Data Folder:", val_data_folder)
    if train_data_folder is None or val_data_folder is None:
        print("Error: Train or validation data folder not found.")
        return
         
        for file_path in file_paths:
            #Perform note detection on each audio file
            detected_note = note_detect(file_path)
            # Print detected note
            print("Detected Note:", detected_note)
            plot_audio_subset(file_path, subset_size=1000)
        print('done')
        
    
    for subfolder_name in subfolders:
        if subfolder_name == 'nsynth-train':
            train_data_folder = os.path.join(dataset_path, subfolder_name)
        elif subfolder_name == 'nsynth-valid':
            val_data_folder = os.path.join(dataset_path, subfolder_name)
    
    
    for subfolder_name in subfolders:
            print("Subfolder:", subfolder_name)
            subfolder_path = os.path.join(dataset_path, subfolder_name)
            if subfolder_name == "nsynth-train": 
                rename_files_to_numbers(subfolder_path)
            files, file_paths = list_files_in_folder(subfolder_path)
    train_files, train_file_paths = list_files_in_folder(train_data_folder)
    val_files, val_file_paths = list_files_in_folder(val_data_folder) 

    validation_ratio = 0.2  # 20% of files will be copied to validation

    copy_files_to_validation(train_data_folder, val_data_folder, validation_ratio)
    final_file_path='none'
    for file_path in file_paths:
        final_file_path=file_path

    num_epochs = 5
    batch_size = 100
    input_shape = (16000, 1, 1)
    sample_rate, audio_data = wav.read(final_file_path)
    audio_data = audio_data.reshape(-1, 1)  # Reshape audio data
    num_classes = 88
    model = make_model(input_shape, num_classes)  # Create the model
    
    target_shape = (16000, 1)
    audio_data = load_and_preprocess_audio(file_path, target_shape)
    
    train_generator = custom_data_generator(train_file_paths, batch_size, target_shape)
    val_generator = custom_data_generator(val_file_paths, batch_size, target_shape)
    
    # Use LabelEncoder to convert string labels to integers
    label_encoder = LabelEncoder()

    # Combine train and validation labels
    all_labels = np.concatenate((train_files, val_files))

    # Fit the LabelEncoder on all labels
    label_encoder.fit(all_labels)

    # Transform train and validation labels
    train_labels_encoded = label_encoder.transform(train_files)
    val_labels_encoded = label_encoder.transform(val_files)

       # Use LabelEncoder to convert string labels to integers
    label_encoder = LabelEncoder()
    label_encoder.fit(train_files)  # Fit the LabelEncoder on training labels
    
    # Filter validation data
    val_files_filtered = [file for file in val_files if file in train_files]

    # Transform filtered validation labels
    val_labels_encoded = label_encoder.transform(val_files_filtered)
    train_labels_encoded = label_encoder.transform(train_files)
    all_labels = np.concatenate((train_labels_encoded, val_labels_encoded))
    encoded_labels = label_encoder.fit_transform(all_labels)
    print(encoded_labels)
    
    target_length = 16000  # You can adjust this value as needed
    train_generator = custom_data_generator(train_file_paths, batch_size, target_shape)
    val_generator = custom_data_generator(val_file_paths, batch_size, target_shape)
    train_and_validate_model(model, train_generator, val_generator, num_epochs, batch_size)
    input_data = audio_data.reshape(-1, 32000, 1, 1)  # Reshape input data for the model
    predictions = model.predict(input_data)

if __name__ == "__main__":
    main()
'''
'''
import os
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from data_org import*
from models import*
from audio_processing import*
import pygame

pygame.init()
'''
'''
Path = Find_Path("Notes-Dataset")
subfolders, zipfiles = get_subfolders_and_zipfiles(Path)
#unzip_files(Path, zipfiles)
#for zipfile_name in zipfiles:
    #Path = Find_Path(zipfiles.zipfile_name)
    #extract_to = 'Notes-Dataset'    
    #extract_zip(Path, extract_to) 

for subfolder_name in subfolders:
    subfolder_path = os.path.join(Path, subfolder_name)
    file_paths = list_files_in_folder(subfolder_path)
    
    for file_path in file_paths:
    
        audio_file_path = file_path
        target_shape = (44100, 88200)  # Assuming 44.1 kHz sample rate and 2 seconds of audio
        num_classes = 88  # Replace with the number of classes in your classification task
        train_model(file_path, target_shape, num_classes)
        
        #Detected_Note = note_detect(file_path)
        my_sound = pygame.mixer.Sound(file_path)
        my_sound.play()
        #print("\n\tDetected Note = " + str(Detected_Note))
        


for subfolder_name in subfolders:
    subfolder_path = os.path.join(Path, subfolder_name)
    files,file_paths=list_files_in_folder(subfolder_path)
    #for file in files:
        #note_detect(file)
    for file_path in file_paths:
        
        Detected_Note = note_detect(file_path)
        audio_file = wave.open(file_path)
        Detected_Note = note_detect(audio_file)
        my_sound = pygame.mixer.Sound(file_path)
        my_sound.play()
        print("\n\tDetected Note = " + str(Detected_Note))
        
        #with open(file_path, 'r') as file:
        #freq=find_frequency(file_path)
        #print(freq)
            #audio_file = wave.open(file_paths)
            #Detected_Note = note_detect(audio_file)
            

if __name__ == "__main__":
    main()
'''
