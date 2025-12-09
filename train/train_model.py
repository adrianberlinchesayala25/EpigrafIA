"""
🚀 EpigrafIA - Training Script
==============================
Trains language detection model using Common Voice dataset

Usage:
    python train/train_model.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

import tensorflow as tf
from models.language_model import create_language_model

# ============================================
# Configuration
# ============================================

DATA_DIR = Path("data/Common Voice")
OUTPUT_DIR = Path("outputs/models_trained")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Audio processing params (MUST match predict.py)
SAMPLE_RATE = 16000
DURATION = 3  # seconds
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

# Training params
SAMPLES_PER_LANGUAGE = 500  # Use subset for faster training
BATCH_SIZE = 32
EPOCHS = 50  # More epochs
VALIDATION_SPLIT = 0.2

# Language mapping
LANGUAGES = {
    'Audios Español': 0,
    'Audios Ingles': 1,
    'Audios Frances': 2,
    'Audios Aleman': 3
}

LANGUAGE_NAMES = ['Español', 'Inglés', 'Francés', 'Alemán']


# ============================================
# Data Augmentation
# ============================================

def augment_audio(y, sr):
    """Apply random augmentation to audio"""
    augmented = []
    
    # Original
    augmented.append(y)
    
    # Add noise
    noise = np.random.randn(len(y)) * 0.005
    augmented.append(y + noise)
    
    # Time stretch (speed up/down slightly)
    if random.random() > 0.5:
        stretch_factor = random.uniform(0.9, 1.1)
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
        # Adjust length
        if len(y_stretched) > len(y):
            y_stretched = y_stretched[:len(y)]
        else:
            y_stretched = np.pad(y_stretched, (0, len(y) - len(y_stretched)))
        augmented.append(y_stretched)
    
    # Pitch shift
    if random.random() > 0.5:
        n_steps = random.uniform(-2, 2)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        augmented.append(y_shifted)
    
    return augmented


# ============================================
# Feature Extraction
# ============================================

def extract_features_from_y(y, sr=SAMPLE_RATE):
    """Extract MFCC features from audio array"""
    # Ensure minimum duration (pad or trim)
    min_samples = sr * DURATION
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)), mode='constant')
    else:
        y = y[:min_samples]
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    # Compute delta features
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Stack features: (n_mfcc * 3, time_frames)
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    
    # Normalize
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    # Transpose to (time_frames, features)
    features = features.T
    
    return features


def extract_features(audio_path: str, augment: bool = True) -> list:
    """
    Extract MFCC features from audio file with optional augmentation
    
    Returns:
        List of numpy arrays of shape (time_frames, 120)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        if augment:
            # Get augmented versions
            audio_versions = augment_audio(y, sr)
        else:
            audio_versions = [y]
        
        features_list = []
        for audio in audio_versions:
            features = extract_features_from_y(audio, sr)
            if features is not None:
                features_list.append(features)
        
        return features_list
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return []


# ============================================
# Data Loading
# ============================================

def load_dataset():
    """Load and preprocess dataset with augmentation"""
    
    print("\n" + "="*60)
    print("📂 LOADING DATASET (with augmentation)")
    print("="*60)
    
    X = []
    y = []
    
    for lang_folder, label in LANGUAGES.items():
        lang_path = DATA_DIR / lang_folder / "clips"
        
        if not lang_path.exists():
            print(f"⚠️  Folder not found: {lang_path}")
            continue
        
        # Get audio files
        audio_files = list(lang_path.glob("*.mp3"))
        
        # Shuffle and take subset
        random.shuffle(audio_files)
        audio_files = audio_files[:SAMPLES_PER_LANGUAGE]
        
        print(f"\n📁 {LANGUAGE_NAMES[label]}: {len(audio_files)} files")
        
        count = 0
        # Extract features with augmentation
        for audio_file in tqdm(audio_files, desc=f"   Processing"):
            features_list = extract_features(str(audio_file), augment=True)
            for features in features_list:
                X.append(features)
                y.append(label)
                count += 1
        
        print(f"   📊 Total samples after augmentation: {count}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Dataset loaded:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Classes: {np.bincount(y)}")
    
    return X, y


# ============================================
# Training
# ============================================

def train():
    """Main training function"""
    
    print("\n" + "="*60)
    print("🚀 EPIGRAFIA LANGUAGE MODEL TRAINING")
    print("="*60)
    
    # Load data
    X, y = load_dataset()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, stratify=y, random_state=42
    )
    
    print(f"\n📊 Train/Val split:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val: {X_val.shape[0]} samples")
    
    # Calculate class weights to balance training
    # Give MORE weight to French and German to improve their detection
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    
    # Boost French (2) and German (3) even more
    class_weights_arr[2] *= 1.5  # French
    class_weights_arr[3] *= 2.0  # German (needs most help)
    
    class_weights = {i: w for i, w in enumerate(class_weights_arr)}
    print(f"\n⚖️ Class weights: {class_weights}")
    
    # Create model
    print("\n🧠 Creating model...")
    input_shape = (X.shape[1], X.shape[2])  # (time_frames, features)
    model = create_language_model(input_shape=input_shape, num_classes=4)
    
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "language_model_best.keras"),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train
    print("\n" + "="*60)
    print("🏋️ TRAINING")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,  # Use class weights to balance!
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("📈 EVALUATION")
    print("="*60)
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save final model
    model.save(str(OUTPUT_DIR / "language_model.keras"))
    print(f"\n✅ Model saved to {OUTPUT_DIR / 'language_model.keras'}")
    
    return model, history


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 GPU detected: {gpus}")
    else:
        print("💻 Running on CPU (this will be slower)")
    
    # Train
    model, history = train()
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
