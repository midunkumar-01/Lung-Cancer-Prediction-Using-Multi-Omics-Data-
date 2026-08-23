import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ✅ Path to dataset
DATASET_PATH = r"C:\Users\Administrator\Desktop\Lung_Disease\datasets"  # Update this with your dataset path

# ✅ Load the CSV file that maps patient IDs to diagnoses
LABELS_FILE = r'C:\Users\Administrator\Desktop\Lung_Disease\patient_diagnosis.csv'  # Replace with your labels CSV file path

# Read the labels CSV file
labels_df = pd.read_csv(LABELS_FILE, header=None, names=['PatientID', 'label'])  # Explicitly define column names

# Check the first few rows to verify the content
print(labels_df.head())

# Create a dictionary mapping PatientID to label for faster lookup
patient_labels = dict(zip(labels_df['PatientID'], labels_df['label']))

# Function to extract labels from the CSV file based on patient ID
def get_label_from_csv(patient_id):
    """
    Extracts the diagnosis label from the CSV file based on patient ID.
    """
    # Check if the patient_id exists in the dictionary
    if patient_id in patient_labels:
        return patient_labels[patient_id]
    else:
        print(f"⚠️ Patient ID {patient_id} not found in the CSV!")
        return None

# Function to extract MFCC features from audio
def extract_features(file_path, max_pad_len=400):
    """
    Extracts MFCC features from the audio file.
    Returns a (400, 40) feature matrix.
    """
    y, sr = librosa.load(file_path, sr=22050)
    
    # Handle empty or corrupted files
    if len(y) < 1:
        print(f"⚠️ Skipping empty file: {file_path}")
        return np.zeros((400, 40))

    # Extract 40 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Pad or truncate to a fixed size (400 time steps)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    # ✅ Normalize MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    return mfccs.T  # Transpose to (400, 40) for CNN compatibility

# ✅ Load dataset and extract features/labels
features, labels = [], []

for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        # Extract the numeric part of the filename (e.g., 101 from 101_1b1_Al_sc_Meditron.wav)
        patient_id = int(file.split('_')[0])  # Get the numeric ID from the filename

        # Get the diagnosis label from the CSV file
        label = get_label_from_csv(patient_id)
        
        if label:  # Only add features if the label is found
            file_path = os.path.join(DATASET_PATH, file)
            features.append(extract_features(file_path))
            labels.append(label)

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

print(f"✅ Dataset Loaded: {X.shape[0]} samples")
print("✅ Unique Labels:", set(labels))

# ✅ Check dataset balance
unique, counts = np.unique(y, return_counts=True)
for lbl, count in zip(unique, counts):
    print(f"Label: {lbl}, Count: {count}")

# ✅ Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Save label encoder for later use
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Fix input shape for CNN
X_train = X_train[..., np.newaxis]  # Shape (samples, 400, 40, 1)
X_test = X_test[..., np.newaxis]

# ✅ CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(400, 40, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')  # Multi-class output
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ Train Model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# ✅ Save Model
model.save("respiratory_model.h5")

print("🎉 Training Complete. Model Saved!")
