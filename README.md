# 🎤 Speech Emotion Recognition

> **Real-time Emotion Detection from Speech using Deep Learning (LSTM)**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 **Overview**

This project implements a **Speech Emotion Recognition (SER)** system that detects human emotions from speech audio in real-time. The system extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features from audio signals and uses a **Long Short-Term Memory (LSTM)** deep learning model to classify emotions.

### 🎯 **Key Features**
- ✅ Real-time emotion detection from microphone input
- ✅ Audio recording and playback functionality
- ✅ Support for 8 different emotions
- ✅ MFCC feature extraction
- ✅ LSTM-based deep learning model
- ✅ Ensemble prediction for better accuracy
- ✅ Audio waveform visualization
- ✅ Model persistence (save/load)

---

## 🧠 **Emotions Detected**

| Emotion | Emoji | Description |
|---------|-------|-------------|
| **Angry** | 😡 | Anger, frustration |
| **Calm** | 😌 | Calm, relaxed |
| **Disgust** | 🤢 | Disgust, dislike |
| **Fear** | 😨 | Fear, scared |
| **Fearful** | 😰 | Anxious, worried |
| **Happy** | 😊 | Joy, happiness |
| **Neutral** | 😐 | Neutral, no emotion |
| **Sad** | 😢 | Sadness, grief |
| **Surprised** | 😲 | Surprise, shock |

---

## 📊 **Dataset Used**

| Dataset | Size | Emotions | Language |
|---------|------|----------|----------|
| **RAVDESS** | 1,440 files | 8 | English |
| **TESS** | 2,800 files | 7 | English |
| **CREMA-D** | 7,442 files | 6 | English |
| **Total** | **11,682 files** | 8 | English |

---

## 🏗️ **Project Architecture**

---┌─────────────────────────────────────────────────────────────┐
│ 1. AUDIO INPUT │
│ (Microphone / WAV File) │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ 2. FEATURE EXTRACTION │
│ • MFCC (20 coefficients) │
│ • Fixed length: 50 time steps │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ 3. LSTM MODEL │
│ • Bidirectional LSTM (128 units) │
│ • Dropout (0.3) │
│ • Dense layer (64 units) │
│ • Softmax output (8 emotions) │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PREDICTION │
│ • Real-time detection │
│ • Ensemble voting (3 trials) │
│ • Confidence score │
└─────────────────────────────────────────────────────────────┘

## 🧠 **Model Architecture**

```python
Model: Sequential
─────────────────────────────────────────────────────────────
Layer (type)              Output Shape         Param #
─────────────────────────────────────────────────────────────
Bidirectional LSTM       (None, 50, 256)       152,576
Dropout                  (None, 50, 256)        0
Bidirectional LSTM       (None, 64)             82,176
Dropout                  (None, 64)             0
Dense                    (None, 64)             4,160
Dropout                  (None, 64)             0
Dense (Softmax)          (None, 8)              520
─────────────────────────────────────────────────────────────
Total params: 239,432
Trainable params: 239,432
Non-trainable params: 0
─────────────────────────────────────────────────────────────


🛠️ Tech Stack
Category	Technologies
Language	Python 3.8+
Deep Learning	TensorFlow / Keras
Audio Processing	Librosa, SoundDevice, SoundFile
Data Processing	NumPy, Pandas
Visualization	Matplotlib, Seaborn
Model Deployment	Joblib, H5 (Keras)


 Installation
1. Clone the Repository
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition

2. Create Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt
4. Download Datasets
Download the following datasets and place them in the data/ directory:

RAVDESS

TESS

CREMA-D
5. Run the Application
bash
python real_time_emotion_detection.py


 Usage
Option 1: Single Recording
python
# Record once and detect emotion
emotion, confidence, filename = real_time_emotion_detection(duration=3)


Option 2: Ensemble Recording
python
# Record 3 times for better accuracy (majority voting)
emotion, avg_confidence = ensemble_emotion_detection(num_trials=3, duration=2)


Option 3: Predict from Existing Audio File
python
def predict_from_file(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    emotion, confidence = predict_emotion(audio)
    print(f"Emotion: {emotion}, Confidence: {confidence:.2f}")

🎯 Real-Time Demo
Run the Application
bash
python real_time_emotion_detection.py
Sample Output
text
============================================================
🎤 REAL-TIME EMOTION DETECTION
============================================================
🔊 Speak clearly into the microphone...
⏱️ Recording duration: 3 seconds
============================================================

🎤 Recording for 3 seconds...
🔴 Speak now...
✅ Recording complete!

📁 Audio saved to: recording_20240615_143022.wav

==================================================
🎯 EMOTION DETECTION RESULT
==================================================
   Emotion: 😊 HAPPY
   Confidence: 78.4%
==================================================



📊 Results Visualization
Confusion Matrix
https://confusion_matrix.png

Training History
https://training_history.png

Audio Waveform
https://audio_waveform.png
