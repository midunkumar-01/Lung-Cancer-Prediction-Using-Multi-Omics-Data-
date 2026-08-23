from django.shortcuts import render
from .forms import AudioFileForm
from .models import AudioFile
import librosa
import tensorflow as tf
import numpy as np
import os
import pickle

# Load the model and label encoder when the server starts
model = tf.keras.models.load_model('respiratory_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Disease recommendations & details
DISEASE_INFO = {
    "COPD": {
        "description": "Chronic Obstructive Pulmonary Disease affects lung airflow, causing difficulty in breathing.",
        "recommendation": "Avoid smoking, use prescribed inhalers, and follow a pulmonary rehabilitation program.",
        "icon": "😷"
    },
    "Pneumonia": {
        "description": "An infection causing lung inflammation, leading to cough, fever, and breathing difficulty.",
        "recommendation": "Take antibiotics, get adequate rest, and stay hydrated.",
        "icon": "🌡"
    },
    "URTI": {
        "description": "Upper Respiratory Tract Infection affects the nose, throat, and sinuses, causing congestion and cough.",
        "recommendation": "Drink warm fluids, use decongestants, and rest well.",
        "icon": "🤧"
    },
    "Healthy": {
        "description": "Your lungs sound normal with no significant issues detected.",
        "recommendation": "Maintain a healthy lifestyle and stay hydrated.",
        "icon": "✅"
    },
    "Bronchiolitis": {
        "description": "Inflammation of the small airways, often due to viral infections, leading to wheezing and difficulty breathing.",
        "recommendation": "Use a humidifier, monitor oxygen levels, and seek medical attention if symptoms worsen.",
        "icon": "🫀"
    },
    "Asthma": {
        "description": "A chronic condition causing airway inflammation and breathing difficulty, often triggered by allergens.",
        "recommendation": "Use prescribed inhalers, avoid triggers, and follow an asthma action plan.",
        "icon": "💨"
    },
    "Bronchiectasis": {
        "description": "A condition where the airways become widened and damaged, leading to chronic coughing and mucus buildup.",
        "recommendation": "Use airway clearance techniques, antibiotics if necessary, and stay active.",
        "icon": "🏥"
    },
    "LRTI": {
        "description": "Lower Respiratory Tract Infection affecting the lungs and bronchial tubes, leading to coughing and fever.",
        "recommendation": "Stay hydrated, use fever reducers, and follow medical advice.",
        "icon": "🩺"
    },
}

# Function to extract MFCC features from the uploaded file
def extract_features(file_path, max_pad_len=400):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    return mfccs.T  # Transpose to (400, 40) for CNN compatibility

# View to handle the file upload and prediction
def upload_audio(request):
    if request.method == 'POST' and request.FILES['file']:
        form = AudioFileForm(request.POST, request.FILES)
        if form.is_valid():
            audio_file = form.save()  # Save the uploaded file

            # Extract features from the uploaded file
            file_path = audio_file.file.path
            features = extract_features(file_path)
            features = np.expand_dims(features, axis=-1)  # Add the channel dimension

            # Make prediction using the model
            prediction = model.predict(np.array([features]))  # Predict for the uploaded file
            predicted_index = np.argmax(prediction)
            predicted_label = le.inverse_transform([predicted_index])[0]
            confidence_score = round(float(prediction[0][predicted_index]) * 100, 2)  # Convert to percentage

            # Get disease details
            disease_info = DISEASE_INFO.get(predicted_label, {
                "description": "No information available.",
                "recommendation": "Consult a doctor for further diagnosis.",
                "icon": "❓"
            })

            # Update the diagnosis in the database
            audio_file.diagnosis = predicted_label
            audio_file.save()

            return render(request, 'upload_success.html', {
                'diagnosis': predicted_label,
                'confidence_score': confidence_score,
                'description': disease_info["description"],
                'recommendation': disease_info["recommendation"],
                'icon': disease_info["icon"]
            })
    else:
        form = AudioFileForm()

    return render(request, 'upload_audio.html', {'form': form})
