import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import warnings
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# Constants
SAMPLE_RATE = 22050
DURATION = 3
MAX_PAD_LEN = 130
N_MFCC = 40
MODEL_PATH = "cnn_emotion_model.h5"
ENCODER_PATH = "label_encoder.pkl"

# Emotion ID to Label Mapping
emotion_map = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
}


# Helper: Get index → (ID, label) mapping
def get_index_to_emotion(le):
    label_to_id = {v: k for k, v in emotion_map.items()}
    index_to_label = {i: label for i, label in enumerate(le.classes_)}
    return {
        i: (label_to_id[label], label)
        for i, label in index_to_label.items()
    }

@st.cache_resource
def load_model_and_encoder():
    model =load_model(MODEL_PATH)
    le =joblib.load(ENCODER_PATH)
    return model, le

# MFCC Extraction
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_norm = mfcc / np.max(np.abs(mfcc))

    if mfcc_norm.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc_norm.shape[1]
        mfcc_norm = np.pad(mfcc_norm, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc_norm = mfcc_norm[:, :MAX_PAD_LEN]

    return mfcc_norm

# Predict Emotion
def predict_emotion(audio_path, model, index_to_emotion):
    mfcc = extract_mfcc_for_test(audio_path)
    mfcc_norm = mfcc / np.max(np.abs(mfcc))
    input_tensor = np.expand_dims(mfcc_norm, axis=-1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    prediction = model.predict(input_tensor)[0]
    predicted_index = np.argmax(prediction)
    emotion_id, emotion_name = index_to_emotion[predicted_index]

    return emotion_id, emotion_name, prediction, mfcc_norm

# Streamlit UI
st.title("Speech Emotion Recognition")
st.write("Upload a `.wav` file to predict the emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    model, le = load_model_and_encoder()
    index_to_emotion = get_index_to_emotion(le)

    # Save uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    emotion_id, emotion_name, prediction, mfcc = predict_emotion("temp.wav", model, index_to_emotion)

    # Display Result
    st.subheader(f"Predicted Emotion: {emotion_name.upper()} (ID: {emotion_id})")

    # Display probabilities
    st.markdown("### Class Probabilities")
    for i, prob in enumerate(prediction):
        eid, label = index_to_emotion[i]
        st.write(f"Class {i}: Emotion ID {eid} → **{label.upper()}** → {prob*100:.2f}%")



# In[5]:






