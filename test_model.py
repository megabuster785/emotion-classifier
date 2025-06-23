import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

# Constants
SAMPLE_RATE = 22050
DURATION = 3
MAX_PAD_LEN = 130
N_MFCC = 40
MODEL_PATH = "../models/cnn_emotion_model.h5"
ENCODER_PATH = "../models/label_encoder.pkl"

# Emotion ID to Label
emotion_map = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
}

# Mapping label index to emotion
def get_index_to_emotion(le):
    label_to_id = {v: k for k, v in emotion_map.items()}
    index_to_label = {i: label for i, label in enumerate(le.classes_)}
    return {
        i: (label_to_id[label], label)
        for i, label in index_to_label.items()
    }

# Load model, encoder, and normalization constant
model = load_model(MODEL_PATH)
le = joblib.load(ENCODER_PATH)
index_to_emotion = get_index_to_emotion(le)
max_val = np.load("../models/max_val.npy")

# Extract MFCC for inference
def extract_mfcc_combined_for_test(audio_path, max_pad_len=130, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc / max_val 
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def pad(x):
        if x.shape[1] < max_pad_len:
            return np.pad(x, ((0, 0), (0, max_pad_len - x.shape[1])), mode='constant')
        else:
            return x[:, :max_pad_len]

    mfcc = pad(mfcc)
    delta = pad(delta)
    delta2 = pad(delta2)

    combined = np.stack([mfcc, delta, delta2], axis=0)
    reshaped = combined.reshape(-1, max_pad_len).T  
    return reshaped, mfcc

# Predict single file
def predict_single(audio_path):
    reshaped, mfcc_vis = extract_mfcc_combined_for_test(audio_path)
    reshaped =  reshaped / max_val  
    input_tensor = reshaped[np.newaxis, ...]  

    prediction = model.predict(input_tensor)[0]
    predicted_index = np.argmax(prediction)
    emotion_id, emotion_name = index_to_emotion[predicted_index]

    print(f"\nPredicted Emotion: {emotion_name.upper()} (ID: {emotion_id})")
    print("\nClass Probabilities:")
    for i, prob in enumerate(prediction):
        eid, label = index_to_emotion[i]
        print(f"Class {i}: Emotion ID {eid} → {label:>10} → {prob*100:.2f}%")

    # Plot MFCC
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc_vis, x_axis='time')
    plt.colorbar()
    plt.title('MFCC (Original - Channel 0)')
    plt.tight_layout()
    plt.show()

# Run prediction
test_file = "../data/Audio_Speech_Actors_01_24/Actor_01/03-01-05-01-01-01-01.wav"
predict_single(test_file)






# In[ ]:




