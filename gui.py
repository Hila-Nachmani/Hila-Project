# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 09:07:41 2024

@author: hilan
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 09:07:41 2024

@author: hilan
"""
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import csv
import random
from models import *
from data_org import *
from audio_processing import *
import tensorflow as tf
import numpy as np

class AudioProcessingApp:
    def __init__(self, root):
        self.root = root
        self.model_path = None 
        self.root.title("Note Detection Project")
        self.root.geometry("600x500")
        
        # Create title and description
        self.title_label = tk.Label(root, text="Note Detection Project", font=("Helvetica", 16))
        self.title_label.pack(pady=20)
        
        self.description_label = tk.Label(root, text="This application allows you to create and run models for audio note detection. "
                                                     "You can test all audio files in a directory or a specific number of files.",
                                          wraplength=500, justify="left")
        self.description_label.pack(pady=10)
        
        self.create_model_button = tk.Button(root, text="Create Model", command=self.create_model)
        self.create_model_button.pack(pady=10)
        
        self.run_model_button = tk.Button(root, text="Load Model", command=self.run_existing_model)
        self.run_model_button.pack(pady=10)
        
        self.train_model_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.upload_detected_notes_button = tk.Button(root, text="Upload Detected Test Notes", command=self.upload_detected_notes)
        self.upload_button = tk.Button(root, text="Upload Label Encoder", command=self.upload_label_encoder)
        self.process_test_data_button = tk.Button(root, text="Process Test Data", command=self.process_test_data)
        self.test_all_button = tk.Button(root, text="Test All Files", command=self.test_all)
        self.test_specific_button = tk.Button(root, text="Test Specific Number of Files", command=self.test_specific_files)

        # Initially hide all buttons except create_model_button and run_model_button
        self.train_model_button.pack_forget()
        self.upload_detected_notes_button.pack_forget()
        self.upload_button.pack_forget()
        self.process_test_data_button.pack_forget()
        self.test_all_button.pack_forget()
        self.test_specific_button.pack_forget()

    def create_model(self):
        print("Create Model button pressed")
        self.train_model_button.pack(pady=10)
        input_shape = (128, 126, 1)  # Define input shape
        num_classes = 88  # Define number of classes
        try:
            self.model = build_model(input_shape, num_classes)
            self.model.summary()
            messagebox.showinfo("Info", "Model created successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create model: {e}")
    '''
    def run_existing_model(self):
            model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("HDF5 files", "*.h5")])
            if model_path:
                self.model = tf.keras.models.load_model(model_path)
                messagebox.showinfo("Info", f"Model loaded from {model_path}")
    '''
    def run_existing_model(self):
        # Ask user to select the model file
        print("Run Existing Model button pressed")
        self.upload_detected_notes_button.pack(pady=10)
        
        self.model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("HDF5 files", "*.h5")])

        if self.model_path:
            try:
                # Load the model from the selected path
                self.model = tf.keras.models.load_model(self.model_path)

                # Explicitly recompile the model to ensure metrics are initialized
                self.model.compile(optimizer=self.model.optimizer,
                                   loss=self.model.loss,
                                   metrics=self.model.metrics)

                # Show info message
                messagebox.showinfo("Info", f"Model loaded from {self.model_path}")

            except Exception as e:
                # Handle any exceptions that occur during loading or compiling
                messagebox.showerror("Error", f"Failed to load or compile model: {e}")
    
    def upload_detected_notes(self):
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        print("Upload Detected Test Notes button pressed")
        self.upload_button.pack(pady=10)
        
        detected_notes_path = filedialog.askopenfilename(title="Upload Detected Test Notes", filetypes=[("CSV files", "*.csv")])
        if detected_notes_path:
            self.detected_test_notes = self.read_detected_notes(detected_notes_path)
            messagebox.showinfo("Info", f"Detected test notes uploaded successfully from {detected_notes_path}")
    '''
    def load_detected_notes(self):
        detected_notes_path = filedialog.askopenfilename(title="Upload Detected Test Notes", filetypes=[("CSV files", "*.csv")])
        if detected_notes_path:
            self.detected_test_notes = detected_notes_path
            messagebox.showinfo("Info", f"Detected test notes uploaded successfully from {detected_notes_path}")
    '''
    def upload_label_encoder(self):
            print("Upload Label Encoder button pressed")
            self.process_test_data_button.pack(pady=10)
             
            file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
            if file_path:
                try:
                    # Load the label encoder from the selected file
                    self.label_encoder = joblib.load(file_path)
                    print("Label encoder loaded successfully!")
                    messagebox.showinfo("Info", f"Label encoder loaded successfully from {file_path}")

                except Exception as e:
                    print(f"Error loading label encoder: {e}")
    
    def train_model(self):
        print("Train Model button pressed")
        
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showerror("Error", "Please create a model first!")
            return

        # Define create_dataset function inside train_model method
        def create_dataset(data_dir, batch_size=32):
            
            def preprocess(file_path):
                audio = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
                spectrogram = tf.signal.stft(tf.squeeze(audio.audio, axis=-1), frame_length=256, frame_step=128)
                spectrogram = tf.abs(spectrogram)
                spectrogram = tf.math.log(spectrogram + 1e-6)
                spectrogram = tf.expand_dims(spectrogram, axis=-1)
                spectrogram = tf.image.resize_with_pad(spectrogram, 128, 126)
                return spectrogram, 0

            list_ds = tf.data.Dataset.list_files(data_dir + '/*.wav')
            dataset = list_ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            return dataset

        train_data_dir = filedialog.askdirectory(title="Select Training Data Directory")
        val_data_dir = filedialog.askdirectory(title="Select Validation Data Directory")
        if train_data_dir and val_data_dir:
            epochs = simpledialog.askinteger("Input", "Enter number of epochs", initialvalue=10)
            batch_size = simpledialog.askinteger("Input", "Enter batch size", initialvalue=32)
            try:
                train_dataset = create_dataset(train_data_dir, batch_size=batch_size)
                val_dataset = create_dataset(val_data_dir, batch_size=batch_size)
                
                self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
                
                # Save the model to the specified directory
                save_dir = os.path.join(os.getcwd(), "ML-Project", "saved_models")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                model_name = "your_model_name.h5"  # Change the model name as needed
                model_path = os.path.join(save_dir, model_name)
                self.model.save(model_path)
                self.model_path = model_path
                
                messagebox.showinfo("Info", "Model trained and saved successfully! please load it to precced")
            except Exception as e:
                messagebox.showerror("Error", "Failed to train model: {}".format(e))

    '''
    def test_all_files(self):
        test_data_dir = filedialog.askdirectory(title="Select Test Data Directory")
        if test_data_dir:
            results = []
            correct_count = 0
            total_count = 0
            for root, _, files in os.walk(test_data_dir):
                for file in files:
                    if file.endswith(".wav"):
                        file_path = os.path.join(root, file)
                        result = self.test_file(file_path)
                        results.append((file, result))
                        if result == "Correct":  # Adjust this condition based on your test result format
                            correct_count += 1
                        total_count += 1

            accuracy_rate = correct_count / total_count if total_count != 0 else 0

            result_window = tk.Toplevel(self.root)
            result_window.title("Test Results")
            result_window.geometry("400x300")

            # Create a scrollable frame
            canvas = tk.Canvas(result_window)
            scroll_y = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(
                    scrollregion=canvas.bbox("all")
                )
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scroll_y.set)

            tk.Label(scrollable_frame, text=f"Accuracy Rate: {accuracy_rate:.2%}").pack()

            for file, result in results:
                tk.Label(scrollable_frame, text=f"{file}: {result}").pack()

            canvas.pack(side="left", fill="both", expand=True)
            scroll_y.pack(side="right", fill="y")
    '''
    
    def test_all(self):
        if not hasattr(self, 'detected_test_notes') or self.detected_test_notes is None:
            messagebox.showerror("Error", "Detected test notes not loaded. Please load the detected test notes first.")
            return
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        # Check if model was created through GUI (without a path)
        if hasattr(self, 'model_path') and self.model_path is None:
            model = self.model  # Use the model object directly if created through GUI
        else:
            model = load_model(self.model_path)  # Load model from specified path
        
        # Extract test data
        test_resized_spectrograms = np.array(self.process_test_data)  
        test_labels = np.array(list(self.detected_test_notes.values()))  # Extracting values from dictionary
        
        # Get the predictions for the test set
        predicted_labels = []
        predicted_notes = []
        for i in tqdm(range(len(test_resized_spectrograms))):
            test_prediction = model.predict(np.expand_dims(test_resized_spectrograms[i], axis=0))
            if isinstance(test_prediction, np.ndarray) and len(test_prediction) > 0:
                predicted_label = np.argmax(test_prediction[0]) 
                predicted_labels.append(predicted_label)
                predicted_note = self.label_encoder.inverse_transform([predicted_label])[0]
                predicted_notes.append(predicted_note)
        
        # Calculate overall accuracy
        accuracy = accuracy_score(test_labels, predicted_notes)
        print("Test Accuracy:", accuracy)
        
        messagebox.showinfo("success", "Finished testing all files.")
        
        # Display results in a new window
        result_window = tk.Toplevel(self.root)
        result_window.title("Test Results")
        result_window.geometry("550x300")
        
        canvas = tk.Canvas(result_window)
        scroll_y = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll_y.set)
        
        for predicted_note in predicted_notes:
            tk.Label(scrollable_frame, text=f"Detected Note: {predicted_note}").pack()

        canvas.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")
        
        # Display accuracy rate
        tk.Label(result_window, text=f"Accuracy Rate: {accuracy:.2%}").pack()

        
    def test_specific_files(self):
        if not hasattr(self, 'detected_test_notes') or self.detected_test_notes is None:
            messagebox.showerror("Error", "Detected test notes not loaded. Please load the detected test notes first.")
            return
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showerror("Error", "Please load or build a model first!")
            return
        if hasattr(self, 'model') and self.model_path is None:
            model = self.model  # Use the model object directly if created through GUI
        else:
            model = load_model(self.model_path)  # Load model from specified path
    
        # Ask for the test data directory
        test_data_dir = filedialog.askdirectory(title="Select Test Data Directory")
        if test_data_dir:
            # Ask for the number of files to test
            num_files = simpledialog.askinteger("Select Number of Files", "Enter the number of files to test (1-766):")
            if num_files:
                # Ensure test data is processed
                if self.process_test_data is None:
                    messagebox.showerror("Error", "Test data not processed. Please process test data first.")
                    return

                # Extract test data and labels
                test_resized_spectrograms = np.array(self.process_test_data[:num_files])
                test_labels = np.array(list(self.detected_test_notes.values()))[:num_files]
                
                # Get the predictions for the test set
                predicted_notes = []
                for i in tqdm(range(len(test_resized_spectrograms))):
                    test_prediction = model.predict(np.expand_dims(test_resized_spectrograms[i], axis=0))
                    if isinstance(test_prediction, np.ndarray) and len(test_prediction) > 0:
                        predicted_label = np.argmax(test_prediction[0])
                        predicted_note = self.label_encoder.inverse_transform([predicted_label])[0]
                        predicted_notes.append(predicted_note)
                    else:
                        predicted_notes.append("Unknown")  # Handle cases where prediction fails

                # Calculate accuracy
                accuracy = accuracy_score(test_labels, predicted_notes)
                print("Test Accuracy:", accuracy)
                
                messagebox.showinfo("Success", f"Finished testing {num_files} files.")

                # Display results in a new window
                result_window = tk.Toplevel(self.root)
                result_window.title("Test Results")
                result_window.geometry("500x300")
                
                canvas = tk.Canvas(result_window)
                scroll_y = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
                scrollable_frame = tk.Frame(canvas)
                
                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(
                        scrollregion=canvas.bbox("all")
                    )
                )
                
                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                canvas.configure(yscrollcommand=scroll_y.set)
                
                for predicted_note in predicted_notes:
                    tk.Label(scrollable_frame, text=f"Detected Note: {predicted_note}").pack()

                canvas.pack(side="left", fill="both", expand=True)
                scroll_y.pack(side="right", fill="y")
                
                # Display accuracy rate
                tk.Label(result_window, text=f"Accuracy Rate: {accuracy:.2%}").pack()

    
    def test_file(self, file_path):
        """
        Predict the musical note for a specific audio file using a pre-trained model.
        
        Parameters:
        - file_path: str, path to the audio file to be tested.
        
        Returns:
        - predicted_note: str, the predicted note for the audio file.
        """
        if self.label_encoder is None:
            print("Label encoder not loaded. Please upload it first.")
            return
        if self.model:
            model = self.model
        else:
            # Load the pre-trained model
            model = joblib.load(self.model_path)
        
        # Load the audio file and convert to spectrogram
        y, sr = librosa.load(file_path, sr=None)
        spectrogram = convert_audio_to_spectrogram(y, sample_rate=sr)
        normalized_spectrogram = normalize_spectrogram(spectrogram)
        resized_spectrogram = resize_spectrogram(normalized_spectrogram, target_shape=(128, 128))
        
        # Reshape spectrogram to match model's input shape (assuming 4D input)
        test_input = resized_spectrogram.reshape(1, resized_spectrogram.shape[0], resized_spectrogram.shape[1], 1)
        
        # Predict the note
        test_prediction = model.predict(test_input)
        predicted_index = np.argmax(test_prediction[0])
        
        # Map the index to the corresponding note name
        note_names = [
            "A0", "A#0", "B0", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1",
            "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2",
            "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3",
            "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4",
            "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5",
            "A5", "A#5", "B5", "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6",
            "A6", "A#6", "B6", "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7",
            "A7", "A#7", "B7", "C8"
        ]
        
        predicted_note = self.label_encoder.inverse_transform([predicted_index])[0]
        print(f"Predicted Note: {predicted_note}")

        # Print the prediction result
        print(f"File: {file_path}, Predicted Note: {predicted_note}")
        
        return predicted_note

    def display_results(self, results, accuracy_rate):
        result_window = tk.Toplevel(self.root)
        result_window.title("Test Results")
        result_window.geometry("400x300")

        # Display accuracy rate
        tk.Label(result_window, text=f"Accuracy Rate: {accuracy_rate:.2%}").pack()

        # Display individual file results
        for file, result in results:
            tk.Label(result_window, text=f"{file}: {result}").pack()

    def read_detected_notes(self, csv_path):
        detected_notes = {}
        with open(csv_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 2:
                    file_path = row[0].strip()  # Assuming file path is in the first column
                    detected_note = row[1].strip()  # Assuming detected note is in the second column
                    detected_notes[file_path] = detected_note
        return detected_notes

    def process_test_data(self):
        print("Process Test Data button pressed")
        self.test_all_button.pack(pady=10)
        self.test_specific_button.pack(pady=10)
        
        test_data_dir = filedialog.askdirectory(title="Select Test Data Directory")
        if test_data_dir:
            try:
                file_paths = sorted([os.path.join(test_data_dir, file) for file in os.listdir(test_data_dir) if file.endswith(".wav")])

                processed_data = []
                for file_path in file_paths:
                    processed_data.append(self.process_single_file(file_path))

                # Now you have processed_data containing spectrograms or other processed data
                # You can store this data or use it for further steps like prediction
                self.process_test_data=processed_data
                messagebox.showinfo("Info", "Test data processing complete.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process test data: {e}")

    def process_single_file(self, file_path):
        try:
            # Load audio data
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            sample_rate=16000
            # Convert audio data to spectrogram
            spectrogram = convert_audio_to_spectrogram(audio_data, sample_rate)

            # Normalize spectrogram
            normalized_spectrogram = normalize_spectrogram(spectrogram)

            # Resize spectrogram to match model input shape
            target_shape = (128, 126)  # Adjust according to your model input shape
            resized_spectrogram = resize_spectrogram(normalized_spectrogram, target_shape)

            # You can return the processed spectrogram or any other processed data
            return resized_spectrogram

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
        
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioProcessingApp(root)
    root.mainloop()
    
    '''
        # Create buttons
        self.create_model_button = tk.Button(root, text="Create Model", command=self.create_model)
        self.create_model_button.pack(pady=10)

        self.run_model_button = tk.Button(root, text="Run Existing Model", command=self.run_existing_model)
        self.run_model_button.pack(pady=10)
        
        self.train_model_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_model_button.pack(pady=10)
        
        self.upload_detected_notes_button = tk.Button(root, text="Upload Detected Test Notes", command=self.upload_detected_notes)
        self.upload_detected_notes_button.pack(pady=10)
        
        self.upload_button = tk.Button(self.root, text="Upload Label Encoder", command=self.upload_label_encoder)
        self.upload_button.pack()
        
        self.process_test_data_button = tk.Button(root, text="Process Test Data", command=self.process_test_data)
        self.process_test_data_button.pack(pady=10)

        self.test_all_button = tk.Button(root, text="Test All Files", command=self.test_all)
        self.test_all_button.pack(pady=10)

        self.test_specific_button = tk.Button(root, text="Test Specific Number of Files", command=self.test_specific_files)
        self.test_specific_button.pack(pady=10)
        '''
        
    '''
    def test_all_files(self):
        if not hasattr(self, 'detected_test_notes') or self.detected_test_notes is None:
            messagebox.showerror("Error", "Detected test notes not loaded. Please load the detected test notes first.")
            return
        
        test_data_dir = filedialog.askdirectory(title="Select Test Data Directory")
        if test_data_dir:
            results = []
            correct_count = 0
            total_count = 0
            
            detected_notes = self.read_detected_notes(self.detected_test_notes)
            print(f"Detected Notes Length: {len(detected_notes)}")  # Check length of detected notes
            
            # List all files in the directory
            file_paths = sorted([os.path.join(test_data_dir, file) for file in os.listdir(test_data_dir) if file.endswith(".wav")])
            detected_note_list = list(detected_notes.keys())
            
            for file_path in file_paths:
                result = self.test_file(file_path)
                results.append(result)
                detected_note = detected_note_list[total_count]
                if result == detected_note:
                    correct_count += 1
                
                total_count += 1
            
            accuracy_rate = correct_count / total_count if total_count != 0 else 0
            print("Test Accuracy:", accuracy_rate)
            
            result_window = tk.Toplevel(self.root)
            result_window.title("Test Results")
            result_window.geometry("400x300")
            
            # Create a scrollable frame
            canvas = tk.Canvas(result_window)
            scroll_y = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(
                    scrollregion=canvas.bbox("all")
                )
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scroll_y.set)
            
            #tk.Label(scrollable_frame, text=f"Accuracy Rate: {accuracy_rate:.2%}").pack()
            
            for result in results:
                tk.Label(scrollable_frame, text=f"Detected Note: {result}").pack()

            canvas.pack(side="left", fill="both", expand=True)
            scroll_y.pack(side="right", fill="y")

    '''
    
    '''
    def test_specific_files(self):
        test_data_dir = filedialog.askdirectory(title="Select Test Data Directory")
        if test_data_dir:
            num_files = simpledialog.askinteger("Input", "Enter number of files to test")
            if num_files:
                files = [f for f in os.listdir(test_data_dir) if f.endswith(".wav")]
                files_to_test = random.sample(files, min(num_files, len(files)))
                results = []
                correct_count = 0
                total_count = 0
                for file in files_to_test:
                    file_path = os.path.join(test_data_dir, file)
                    result = self.test_file(file_path)
                    results.append((file, result))
                    if result == "Correct":  # Adjust this condition based on your test result format
                        correct_count += 1
                    total_count += 1

                accuracy_rate = correct_count / total_count if total_count != 0 else 0

                result_window = tk.Toplevel(self.root)
                result_window.title("Test Results")
                result_window.geometry("400x300")

                # Create a scrollable frame
                canvas = tk.Canvas(result_window)
                scroll_y = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
                scrollable_frame = tk.Frame(canvas)

                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(
                        scrollregion=canvas.bbox("all")
                    )
                )

                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                canvas.configure(yscrollcommand=scroll_y.set)

                tk.Label(scrollable_frame, text=f"Accuracy Rate: {accuracy_rate:.2%}").pack()

                for file, result in results:
                    tk.Label(scrollable_frame, text=f"{file}: {result}").pack()

                canvas.pack(side="left", fill="both", expand=True)
                scroll_y.pack(side="right", fill="y")        
    '''
    '''
    def test_specific_files(self):
        if not hasattr(self, 'detected_test_notes') or self.detected_test_notes is None:
            messagebox.showerror("Error", "Detected test notes not loaded. Please load the detected test notes first.")
            return
        
        test_data_dir = filedialog.askdirectory(title="Select Test Data Directory")
        if test_data_dir:
            num_files = simpledialog.askinteger("Select Number of Files", "Enter the number of files to test (1-766):")
            if num_files:
                results = []
                correct_count = 0
                total_count = 0
                
                detected_notes = self.read_detected_notes(self.detected_test_notes)
                print(f"Detected Notes Length: {len(detected_notes)}")  # Check length of detected notes
                
                # List all files in the directory
                file_paths = sorted([os.path.join(test_data_dir, file) for file in os.listdir(test_data_dir) if file.endswith(".wav")])
                detected_note_list = list(detected_notes.keys())
                
                for file_path in file_paths[:num_files]:
                    result = self.test_file(file_path)
                    results.append(result)
                    detected_note=detected_note_list[total_count]
                    if result == detected_note:
                        correct_count+=1
                    #results.append((os.path.basename(file_path), result))
                    
                    # Extract file name from path to use as key in detected_notes dictionary
                    #file_name = os.path.basename(file_path)
                    
                    # Compare with detected notes
                    #detected_note = detected_notes.get(file_name,None )  # Use .get() to safely access dictionary
                    #print(f"File: {file_name}, Result: {result}, Detected Note: {detected_note}")
                    
                    #if detected_note is not None and result == detected_note:
                        correct_count += 1
                    
                    total_count += 1
                accuracy = accuracy_score(detected_note_list, results)  
                print("Test Accuracy:", accuracy)
                #accuracy_rate = correct_count / total_count if total_count != 0 else 0
                
                result_window = tk.Toplevel(self.root)
                result_window.title("Test Results")
                result_window.geometry("400x300")
                
                # Create a scrollable frame
                canvas = tk.Canvas(result_window)
                scroll_y = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
                scrollable_frame = tk.Frame(canvas)
                
                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(
                        scrollregion=canvas.bbox("all")
                    )
                )
                
                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                canvas.configure(yscrollcommand=scroll_y.set)
                
                tk.Label(scrollable_frame, text=f"Accuracy Rate: {accuracy:.2%}").pack()
                
                for file, result in results:
                    tk.Label(scrollable_frame, text=f"{file}: {result}").pack()
                
                canvas.pack(side="left", fill="both", expand=True)
                scroll_y.pack(side="right", fill="y")
    '''
    '''
    def test_specific_files(self):
        if not hasattr(self, 'detected_test_notes') or self.detected_test_notes is None:
            messagebox.showerror("Error", "Detected test notes not loaded. Please load the detected test notes first.")
            return

        test_data_dir = filedialog.askdirectory(title="Select Test Data Directory")
        if test_data_dir:
            num_files = simpledialog.askinteger("Select Number of Files", "Enter the number of files to test (1-766):")
            if num_files:
                results = []
                correct_count = 0
                total_count = 0

                detected_notes = self.read_detected_notes(self.detected_test_notes)
                print(f"Length of detected notes: {len(detected_notes)}")  # Debugging

                # List all files in the directory
                file_paths = sorted([os.path.join(test_data_dir, file) for file in os.listdir(test_data_dir) if file.endswith(".wav")])

                for file_path in file_paths[:num_files]:
                    result = self.test_file(file_path)
                    results.append((os.path.basename(file_path), result))

                    # Compare with detected notes
                    file_name = os.path.basename(file_path)
                    if file_name in detected_notes:
                        detected_note = detected_notes[file_name]
                        if result == detected_note:
                            correct_count += 1

                    total_count += 1

                accuracy_rate = correct_count / total_count if total_count != 0 else 0

                result_window = tk.Toplevel(self.root)
                result_window.title("Test Results")
                result_window.geometry("400x300")

                # Create a scrollable frame
                canvas = tk.Canvas(result_window)
                scroll_y = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
                scrollable_frame = tk.Frame(canvas)

                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(
                        scrollregion=canvas.bbox("all")
                    )
                )

                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                canvas.configure(yscrollcommand=scroll_y.set)

                tk.Label(scrollable_frame, text=f"Accuracy Rate: {accuracy_rate:.2%}").pack()

                for file, result in results:
                    tk.Label(scrollable_frame, text=f"{file}: {result}").pack()

                canvas.pack(side="left", fill="both", expand=True)
                scroll_y.pack(side="right", fill="y")
    '''
    


'''
import tkinter as tk
from tkinter import filedialog, messagebox
from data_org import *
from models import *
from main import *
from audio_processing import *
import os
import threading

def find_path(target_folder_name):
    for root, dirs, files in os.walk("/"):  # Starting from root directory
        if target_folder_name in dirs:
            return os.path.join(root, target_folder_name)
    return None

class CNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Welcome to music detect")  # Title added here
       
        # Directory selection
        self.dir_label = tk.Label(root, text="Select Dataset Directory:")
        self.dir_label.pack()
        self.dir_button = tk.Button(root, text="Browse", command=self.browse_directory)
        self.dir_button.pack()
        self.dir_path = tk.StringVar()
        self.dir_entry = tk.Entry(root, textvariable=self.dir_path, width=50)
        self.dir_entry.pack()
       
        # Start processing and training
        self.process_button = tk.Button(root, text="Start Processing and Training", command=self.start_processing_and_training)
        self.process_button.pack()
       
        # Test button
        self.test_button = tk.Button(root, text="Run Test", command=self.run_test)
        self.test_button.pack()
       
        # Progress and results display
        self.progress_text = tk.Text(root, height=15, width=80)
        self.progress_text.pack()

    def browse_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.dir_path.set(dir_path)
           
    def start_processing_and_training(self):
        dataset_path = self.dir_path.get()
        if not os.path.isdir(dataset_path):
            messagebox.showerror("Error", "Please select a valid directory")
            return
       
        self.progress_text.insert(tk.END, f"Selected Directory: {dataset_path}\n")
        # Add your processing and training logic here

    def run_test(self):
        dataset_path = self.dir_path.get()
        if not os.path.isdir(dataset_path):
            messagebox.showerror("Error", "Please select a valid directory")
            return
       
        self.progress_text.insert(tk.END, "Starting Test...\n")
        # Add your test logic here

if __name__ == "__main__":
    root = tk.Tk()
    app = CNNApp(root)
    root.mainloop()
'''
