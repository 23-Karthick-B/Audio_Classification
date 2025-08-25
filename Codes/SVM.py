# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 19:11:45 2025

@author: NIVAS G
"""

import os
import sys
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, roc_curve, auc)

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 🔍 1. Dataset Loading (MODIFIED FOR ESC-50) ---

# IMPORTANT: Update this path to where you extracted the dataset
dataset_path = 'E:\Semester 5\P1_AudioClassification\ESC-50-master' # e.g., 'C:/Users/YourUser/Downloads/ESC-50-master'
metadata_path = os.path.join(dataset_path, 'meta/esc50.csv')
audio_path = os.path.join(dataset_path, 'audio')

# Load metadata from the CSV file
try:
    metadata = pd.read_csv(metadata_path)
except FileNotFoundError:
    print(f"Error: Metadata file not found at {metadata_path}")
    print("Please download the ESC-50 dataset and update the 'dataset_path' variable.")
    sys.exit()

print("ESC-50 metadata loaded successfully.")

# --- 🎧 2. MFCC Feature Extraction and Preprocessing ---

def extract_features(file_path):
    """
    Extracts MFCCs and computes their statistics (mean, variance, rms)
    to create a single feature vector for an audio file.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=5)

        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # MFCC Extraction (all steps handled by librosa)
        mfccs = librosa.feature.mfcc(y=emphasized_audio, sr=sample_rate, n_mfcc=13)
        
        # Dimensionality Reduction: Compute statistics
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_var = np.var(mfccs, axis=1)
        
        rms = librosa.feature.rms(y=audio).mean()
        
        feature_vector = np.hstack((mfccs_mean, mfccs_var, rms))
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process all audio files and extract features
features = []
labels = []
# Create binary label: 1 for 'siren', 0 for all other classes
metadata['label'] = metadata['category'].apply(lambda x: 1 if x == 'siren' else 0)


print("\nStarting MFCC feature extraction... (This will be quick)")
for index, row in metadata.iterrows():
    # Construct the full path to the audio file
    file_path = os.path.join(audio_path, row['filename'])
    data = extract_features(file_path)
    if data is not None:
        features.append(data)
        labels.append(row['label'])

print("Feature extraction complete.")

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# --- 🧠 3. SVM Model Training ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features - very important for SVM!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining data shape: {X_train_scaled.shape}")
print(f"Testing data shape: {X_test_scaled.shape}")

# Find best hyperparameters (C, gamma) using GridSearchCV
# We add class_weight='balanced' to handle the extreme imbalance in ESC-50
print("\nTuning hyperparameters for SVM...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train_scaled, y_train)

print("\nBest hyperparameters found:", grid.best_params_)

# Get the best model from the grid search
svm_model = grid.best_estimator_
y_pred = svm_model.predict(X_test_scaled)


# --- 📊 4. Analysis and Evaluation ---

print("\n--- Model Evaluation ---")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Sensitivity (Recall): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:(SVM)")
print(classification_report(y_test, y_pred, target_names=['Non-Siren', 'Siren'], zero_division=0))

# Confusion Matrix
print("Generating confusion matrix plot...")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Siren', 'Siren'],
            yticklabels=['Non-Siren', 'Siren'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (SVM on ESC-50)')
plt.show()

# ROC Curve
print("Generating ROC curve plot...")
y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (SVM)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()