Speech Emotion Recognition using CNN and MFCC
This project is a deep learning-based Speech Emotion Recognition (SER) system trained on the "RAVDESS dataset", using **MFCC features** as input to a "Convolutional Neural Network (CNN)" model. The goal is to classify emotions such as *happy, sad, angry, fearful, surprised, neutral, calm, and disgust* from speech audio.
Project Overview

- Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Classes: 8 emotion categories
- Frameworks Used: TensorFlow/Keras, Librosa, scikit-learn, NumPy, Matplotlib
- Model Type: 2D CNN trained on MFCC features
- Model Output: Saved as "cnn_emotion_model.h5"
- Label Encoder: Saved as "label_encoder.pkl"

Preprocessing Pipeline

1. Audio Loading & MFCC Extraction:
   - Extracted 40 MFCC features per frame from audio using librosa.
   - Each MFCC array is shaped as (40, 130) representing time vs frequency coefficients.
   - Data was normalized to [0, 1] range and reshaped to (40, 130, 1).

2. Data Augmentation:
   - Augmented audio to improve robustness. I did pitch shift,time strectch and time shift.

3. Label Encoding:
   - Emotion labels were encoded using "LabelEncoder", then one-hot encoded.
   - Encoder saved for inference reuse.

4. Train/Validation Split:
   - Stratified 80/20 split on encoded labels for balanced class distribution.

Model Architecture (CNN)
Input: (40, 130, 1) MFCC feature map

→ Conv2D (32 filters) + ReLU + MaxPooling2D + BatchNormalization  
→ Conv2D (64 filters) + ReLU + MaxPooling2D + BatchNormalization  
→ Conv2D (128 filters) + ReLU + MaxPooling2D + BatchNormalization  
→ Flatten  
→ Dense (128 units) + ReLU + Dropout(0.5)  
→ Output Dense (8 units - Softmax)

Loss: Categorical Crossentropy
Optimizer: Adam
Callbacks:
EarlyStopping on validation accuracy (patience = 5)
ReduceLROnPlateau on validation accuracy (patience = 2, factor = 0.5)
Class Imbalance Handling: Used compute_class_weight to compute and apply class weights during training.

Training Performance
Epochs: 32 (with early stopping)
Batch Size: 32
Validation Accuracy: > 80%
Validation F1 Score: > 80%
Per-Class Accuracy: All classes had accuracy greater than 80%
Weighted F1 Score on Validation Set: 0.8760
CNN Validation Accuracy: 0.8767

Classification Report
Printed at the end of training to show precision, recall, F1-score for each class.
Classification Report:

            precision    recall  f1-score   support
  angry       0.89      0.95      0.92       301
  calm        0.87      0.93      0.90       301
  disgust     0.87      0.87      0.87       154
  fearful     0.86      0.81      0.83       301
  happy       0.90      0.89      0.90       301
  neutral     0.88      0.85      0.87       150
  sad         0.83      0.80      0.82       301
  surprised   0.93      0.90      0.91       153
  accuracy                        0.88      1962
  macro avg    0.88      0.88     0.88      1962
weighted avg   0.88      0.88     0.88      1962

Training History
Training and validation accuracy/loss curves were plotted for visualization of model learning.
![image](https://github.com/user-attachments/assets/404e08d8-4085-4598-91fe-ad5310a7d3cc)

 Model Artifacts
Trained Model: cnn_emotion_model.h5
Label Encoder: label_encoder.pkl

How to Run
Extract Features (MFCC):
Run feature extraction script to generate .npz
This step can be skipped if the file is already generated.
Train model:-
run train_cnn_model.ipynb
Note
Large Feature File: cnn_features_augmented.npz is not included due to size. You must run the feature extraction code to generate it before training or inference.
you have to set the file path correctly in extract_cnn_features file.

to do the testing:-
Run test_model.py 
Note:- add ur .wav file in the path file to test the model

to run streamlit app:-
run streamlit run app.py on ur terminal inside the folder







