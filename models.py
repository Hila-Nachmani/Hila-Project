# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 18:54:06 2024

@author: hilan
"""
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import joblib 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

#from keras.models import Model
#import evaluate

CHECKPOINT_FILE_NAME = "mycheckpoint"  # object where the weights are kept
label_encoder = LabelEncoder()
'''
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
'''

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),  # Add dropout after the first pooling layer
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),  
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),  
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),  
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5), 
        Dense(num_classes, activation='softmax')
    ])


    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def train_and_valid(model, train_data, val_data, batch_size, epochs):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    train_X = np.array(train_data['resized_spectrograms'])  # Convert to numpy array
    train_Y = np.array(train_data['closest_notes'])  # Convert to numpy array

    val_X = np.array(val_data['resized_spectrograms'])  # Convert to numpy array
    val_Y = np.array(val_data['closest_notes'])  # Convert to numpy array
    
    # Encode the labels
    all_labels = train_data['closest_notes'] + val_data['closest_notes']
    label_encoder.fit(all_labels)
    train_Y = label_encoder.fit_transform(train_Y)
    val_Y = label_encoder.transform(val_Y)
    print("Starting training...")
    history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_data=(val_X, val_Y),callbacks=[early_stopping])
    
    # Plot training history (loss and accuracy)
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    '''
    plt.savefig('loss_accuracy_valid.png')  # Save the plot
    '''
    # Save the plot
    project_folder = os.path.dirname(os.path.abspath(__file__))
    plot_folder = os.path.join(project_folder, "train_valid_plots")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plot_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_name = f"plot_{plot_timestamp}.png"
    plot_path = os.path.join(plot_folder, plot_name)
    plt.savefig(plot_path)  
    
    '''
    # Display the saved plots
    loss_accuracy_plot = plt.imread('loss_accuracy_valid.png')
    plt.figure(figsize=(10, 5))
    plt.imshow(loss_accuracy_plot)
    plt.axis('off')  # Turn off axis
    plt.show()
    '''
    
    # Save the model
    model_folder = os.path.join(project_folder, "saved_models")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    #model_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"model_{timestamp}.h5"
    model_path = os.path.join(model_folder, model_name)
    model.save(model_path)  # Save the model as an h5 file
    
    # Save the encoder
    encoder_folder = os.path.join(project_folder, "saved_encoders")
    if not os.path.exists(encoder_folder):
        os.makedirs(encoder_folder)

    #encoder_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    encoder_name = f"encoder_{timestamp}.pkl"
    encoder_path = os.path.join(encoder_folder, encoder_name)
    joblib.dump(label_encoder, encoder_path)
    
    #save the weights
    weights_folder = os.path.join(project_folder, "saved_weights")
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    # Create the checkpoint filename
    
    CHECKPOINT_FILE_NAME = f"mycheckpoint_{timestamp}.weights.h5"
    checkpoint_path = os.path.join(weights_folder, CHECKPOINT_FILE_NAME)

    model.save_weights(checkpoint_path)
    
    print("Training complete.")
    
    
    return model_path, history, encoder_path , checkpoint_path
    
def test_model(test_data, model_path, label_encoder, print_predictions=False):
    # Load the model
    model = load_model(model_path)
    
    # Extract test data
    test_resized_spectrograms = np.array(test_data['resized_spectrograms'])  
    test_labels = np.array(test_data['closest_notes'])  
    
    # Get the predictions for the test set
    predicted_labels = []
    predicted_notes = []
    for i in tqdm(range(len(test_resized_spectrograms))):
        test_prediction = model.predict(np.expand_dims(test_resized_spectrograms[i], axis=0))
        if isinstance(test_prediction, np.ndarray) and len(test_prediction) > 0:
            predicted_label = np.argmax(test_prediction[0]) 
            predicted_labels.append(predicted_label)
            predicted_note = label_encoder.inverse_transform([predicted_label])[0]
            predicted_notes.append(predicted_note)
            if print_predictions:
                print("note:", i)
                print("Prediction:", predicted_note)
    '''
    # Print true labels and predicted notes
    for true_label, predicted_note in zip(test_labels, predicted_notes):
        print("True Label:", true_label, "Predicted Note:", predicted_note)
    '''
       
    # Calculate overall accuracy
    accuracy = accuracy_score(test_labels, predicted_notes)  
    print("Test Accuracy:", accuracy)
       
    # Decode true labels
    #true_notes = label_encoder.inverse_transform(test_labels)
 
    # Calculate accuracy per class
    unique_classes = np.unique(test_labels)
    accuracy_per_class = {}
    for true_note, predicted_note in zip(test_labels, predicted_notes):
        if true_note not in accuracy_per_class:
            accuracy_per_class[true_note] = {'true': 0, 'predicted': 0}
        accuracy_per_class[true_note]['true'] += 1
        if true_note == predicted_note:
            accuracy_per_class[true_note]['predicted'] += 1
    
    # Print accuracy per class
    for note, acc in accuracy_per_class.items():
        acc_rate = acc['predicted'] / acc['true']
        print(f"Accuracy for {note}: {acc_rate}")


    # Plot the test accuracy and class-wise accuracy
    fig, axs = plt.subplots(1, 2, figsize=(50, 20))  # Increase the figure width

    # Test accuracy
    axs[0].plot([1], [accuracy], 'ro',markersize=20, label='Test accuracy')
    axs[0].set_title('Test Accuracy',size=30)
    axs[0].set_xlabel('Epochs',size=30)
    axs[0].set_ylabel('Accuracy',size=30)
    #axs[0].tick_params(axis='x',size=25)
    #axs[0].tick_params(axis='y',size=25)
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[0].legend()

    # Class-wise accuracy
    class_labels = list(accuracy_per_class.keys())
    accuracy_values = [acc['predicted'] / acc['true'] for acc in accuracy_per_class.values()]
    axs[1].bar(class_labels, accuracy_values, width=0.9)  # Increase the width of the bars
    axs[1].set_title('Class-wise Accuracy',size=30)
    axs[1].set_xlabel('Class',size=30)
    axs[1].set_ylabel('Accuracy',size=30)
    axs[1].tick_params(axis='x', rotation=70)
    axs[1].tick_params(axis='y', labelsize=12)
    #axs[1].tick_params(axis='y',size=25)
    
    # Save the plot
    project_folder = os.path.dirname(os.path.abspath(__file__))
    plot_folder = os.path.join(project_folder, "test_plots")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plot_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_name = f"plot_{plot_timestamp}.png"
    plot_path = os.path.join(plot_folder, plot_name)
    plt.savefig(plot_path)
    
    plt.tight_layout()
    plt.show()
    




    
'''    
def test_model(test_data, model_path, print_predictions=False):
    
    # load the data
    model = load_model(model_path)
    test_resized_spectrograms = np.array(test_data['resized_spectrograms'])  # Convert to numpy array
    test_labels = np.array(test_data['closest_notes'])  # Convert to numpy array

    # Get the predictions for the test set
    predicted_labels = []
    predicted_notes = []
    for i in tqdm(range(len(test_resized_spectrograms))):
        test_prediction = model.predict(np.expand_dims(test_resized_spectrograms[i], axis=0))
        print(test_prediction)
        if isinstance(test_prediction, np.ndarray) and len(test_prediction) > 0:  # Check if test_prediction is an array and not empty
            predicted_label = np.argmax(test_prediction[0]) # Accessing the first element of the array
            predicted_note = note_names[predicted_label]
            predicted_labels.append(predicted_label)
            predicted_notes.append(predicted_note)
            if print_predictions:
                print("note:", i)
                print("Prediction:", predicted_note)
    
    for true_label, predicted_note in zip(test_labels, predicted_notes):
        print("True Label:", true_label, "Predicted Note:", predicted_note)

    # Calculate overall accuracy
    accuracy = accuracy_score(test_labels, predicted_labels)  # Use predicted_labels here
    print("Test Accuracy:", accuracy)
    
    # Calculate accuracy per class
    unique_classes = np.unique(test_labels)
    accuracy_per_class = []
    for class_label in unique_classes:
        class_true_labels = (test_labels == class_label)
        print("Class true labels shape:", class_true_labels.shape)
        class_pred_notes = (predicted_notes == class_label)
        class_pred_labels = np.array(class_pred_notes).astype(int)  # Convert boolean array to integer labels
        print("Class predicted labels shape:", class_pred_labels.shape)
        class_accuracy = accuracy_score(class_true_labels, class_pred_labels)
        accuracy_per_class.append(class_accuracy)
        


    # Plot the test accuracy and class-wise accuracy
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].plot([1], [accuracy], 'ro', label='Test accuracy')
    axs[0].set_title('Test Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].bar(unique_classes, accuracy_per_class)
    axs[1].set_title('Class-wise Accuracy')
    axs[1].set_xlabel('Class')
    axs[1].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()
    plt.savefig('test_plot.png')
'''




'''
def train_and_plot(model, train_data, val_data, batch_size, epochs):
    # Extract training and validation data
    
    num_channels = 1
    train_X,train_labels = (train_data['resized_spectrograms'][0].shape[0], train_data['resized_spectrograms'][0].shape[1], num_channels),train_data['closest_notes']
    val_X,val_labels = (val_data['resized_spectrograms'][0].shape[0], train_data['resized_spectrograms'][0].shape[1], num_channels),val_data['closest_notes']
    
    train_X, train_labels = train_data['spectrograms'], train_data['closest_notes']
    val_X, val_labels = val_data['spectrograms'], val_data['closest_notes']
    
    train_X = np.array(train_X)
    val_X = np.array(val_X)
    
    # Encode the labels
    all_labels = train_data['closest_notes'] + val_data['closest_notes']
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    train_Y = label_encoder.fit_transform(train_labels)
    val_Y = label_encoder.transform(val_labels)
    
    print("Train X shape:", train_X.shape)
    print("Val X shape:", val_X.shape)
    
    val_X = np.expand_dims(val_X, axis=-1)
    
    # Train the model
    history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_data=(val_X, val_Y))
    
    # Plot training history (loss and accuracy)
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('loss_accuracy_plot.png')  # Save the plot

    # Display the saved plots
    loss_accuracy_plot = plt.imread('loss_accuracy_plot.png')
    plt.figure(figsize=(10, 5))
    plt.imshow(loss_accuracy_plot)
    plt.axis('off')  # Turn off axis
    plt.show()

'''











'''
def train_and_validate_model(model, train_data, val_data, num_epochs, batch_size):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_data:
            with tf.GradientTape() as tape:
                train_predictions = model(batch_data)
                train_loss = loss_fn(batch_labels, train_predictions)

            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss, train_accuracy = evaluate_model(model, train_data, loss_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = evaluate_model(model, val_data, loss_fn)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Train Acc: {train_accuracy}, Val Loss: {val_loss}, Val Acc: {val_accuracy}")

    plot_metrics(train_losses, val_losses, "Loss", "Training and Validation Loss")
    plot_metrics(train_accuracies, val_accuracies, "Accuracy", "Training and Validation Accuracy")


def train_and_validate_model(model, train_generator, val_generator, num_epochs, batch_size):
    loss_fn = tf.keras.losses.sparse_categorical_crossentropy  # Define the loss function
    
    for epoch in range(num_epochs):
        print("Epoch:", epoch + 1)
        train_data = train_generator
        val_data = val_generator
        train_loss = 0.0
        val_loss = 0.0
        
        # Training loop
        for batch_data, batch_labels in train_data:
            print("Batch Labels:", batch_labels)
            train_predictions = model.predict(batch_data)
            print("Train Predictions:", train_predictions)
            
            # Calculate loss
            train_loss += tf.reduce_mean(loss_fn(batch_labels, train_predictions))  # Use tf.reduce_mean to compute the mean loss
        
        # Validation loop
        for batch_data, batch_labels in val_data:
            val_predictions = model.predict(batch_data)
            
            # Calculate loss
            val_loss += tf.reduce_mean(loss_fn(batch_labels, val_predictions))  # Use tf.reduce_mean to compute the mean loss
        
        # Average loss over batches
        train_loss /= len(train_data)
        val_loss /= len(val_data)
        
        print("Train Loss:", train_loss.numpy())
        print("Validation Loss:", val_loss.numpy())



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

def make_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Specify the loss function
                  metrics=['accuracy'])  # Specify metrics for evaluation
    
    return model
'''
'''
def make_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        # Fully connected layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
'''
'''
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf

def make_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        # Fully connected layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
import numpy as np
import scipy.io.wavfile as wav

def load_and_preprocess_audio(audio_file_path, target_shape):
    # Read the audio file
    sample_rate, audio_data = wav.read(audio_file_path)
    
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
'''
