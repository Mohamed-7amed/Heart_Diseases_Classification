import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gradio as gr

# ====================
# Preprocessing Module
# ====================

def load_audio_files(data_path, sample_rate=16000):
    audio_data = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(os.listdir(data_path))}
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            if os.path.isfile(file_path):
                try:
                    audio, sr = sf.read(file_path)
                    if sr != sample_rate:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                    audio_data.append(audio)
                    labels.append(label_map[label])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return audio_data, labels

# We Remove Normalization 

# Reduce noise by separating harmonic components Output ==> [Harmonic , Percussive]
def reduce_noise(audio_data):
    return [librosa.effects.hpss(audio)[0] for audio in audio_data]


# Reshape data to include a channel dimension new shape ==> Shape: (num_samples, height, width, 1)
def reshape_data_for_cnn(X):
    return np.expand_dims(X, axis=-1) 


# ====================
# Feature Extraction Module
# ====================

def extract_mel_spectrograms(audio_data, sample_rate=16000, n_mels=128, max_len=300):
    mel_spectrograms = []
    for audio in audio_data:
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Pad or truncate to max_len
        if mel_spectrogram_db.shape[1] < max_len:
            pad_width = max_len - mel_spectrogram_db.shape[1]
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spectrogram_db = mel_spectrogram_db[:, :max_len]
        mel_spectrograms.append(mel_spectrogram_db)
    return mel_spectrograms

# ====================
# Build Model
# ====================

def create_cnn_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Convolutional blocks
    # First block
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Second block
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Third block
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Fourth block
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Fifth block
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Fully connected layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ====================
# Data Preparation
# ====================


dataset_path = r'./Dataset'

audio_data, labels = load_audio_files(dataset_path)
audio_data = reduce_noise(audio_data)
y = np.array(labels)

# Extract features
mel_spectrograms = extract_mel_spectrograms(audio_data)
X = np.array(mel_spectrograms)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = reshape_data_for_cnn(X_train)
X_test = reshape_data_for_cnn(X_test)

# Initialize and train the model
cnn_input_shape = (X_train.shape[1], X_train.shape[2], 1)
num_classes = len(np.unique(y))
cnn_model = create_cnn_model(cnn_input_shape, num_classes)

print("\nTraining CNN Model:")
history_cnn = cnn_model.fit(
    X_train, y_train, epochs=10, validation_split=0.2, batch_size=32, 
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Evaluate the model
test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
print(f"CNN Test Accuracy: {test_acc * 100:.2f}%")

# ====================
# Graphs
# ====================

# Training and validation accuracy/loss Graphs
def plot_training_validation_metrics(history):
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    image_path = "training_metrics.png"
    plt.savefig(image_path)
    plt.close()
    return image_path

# Confusion matrix
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    image_path = "confusion_matrix.png"
    plt.savefig(image_path)
    plt.close()
    return image_path

# ====================
# Gradio Integration
# ====================

def classify_audio(audio_file):
    y, sr = librosa.load(audio_file, duration=10.0)
    mel_spectrogram = extract_mel_spectrograms([y])[0]
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)

    prediction = cnn_model.predict(mel_spectrogram)
    predicted_class = np.argmax(prediction)
    
    label_map = {label: idx for idx, label in enumerate(os.listdir(dataset_path))}
    inverse_label_map = {v: k for k, v in label_map.items()}
    predicted_label = inverse_label_map[predicted_class]

    # Generate visualizations
    spectrogram_image_path = "spectrogram.png"
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        np.squeeze(mel_spectrogram[0]), 
        sr=sr, 
        x_axis='time', 
        y_axis='mel', 
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(spectrogram_image_path)
    plt.close()

    # Generate training metrics and confusion matrix
    metrics_image_path = plot_training_validation_metrics(history_cnn.history)
    confusion_matrix_image_path = plot_confusion_matrix(cnn_model, X_test, y_test)

    return (
        f"Predicted Label: {predicted_label}", 
        f"Prediction Probability: {(prediction[0][predicted_class]) * 100:.2f}%", 
        spectrogram_image_path, 
        metrics_image_path, 
        confusion_matrix_image_path
    )

# Set up Gradio interface
interface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(sources="upload", type="filepath", label="Upload Heartbeat Audio"),
    outputs=[
        gr.Textbox(label="Predicted Class Label"),
        gr.Textbox(label="Prediction Probability"),
        gr.Image(label="Mel Spectrogram"),
        gr.Image(label="Training/Validation Metrics"),
        gr.Image(label="Confusion Matrix")
    ],
    title="Heartbeat Audio Classifier",
    description="Upload an audio file (up to 10 seconds) and classify the heartbeat sound while displaying various visualizations."
)

interface.launch()
