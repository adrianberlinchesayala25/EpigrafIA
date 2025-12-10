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
SAMPLES_PER_LANGUAGE = 1000  # More samples for better accuracy
BATCH_SIZE = 32
EPOCHS = 80  # More epochs for convergence
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

def augment_audio(y, sr, extra_augmentation=False):
    """
    Apply augmentation to simulate different recording conditions.
    This helps the model generalize to real-time microphone recordings.
    
    Args:
        y: audio signal
        sr: sample rate
        extra_augmentation: if True, apply more augmentations (for underrepresented classes)
    """
    augmented = []
    
    # Original (always included)
    augmented.append(y)
    
    # 1. Add background noise (simulates different environments)
    if random.random() > 0.3:
        noise_level = random.uniform(0.001, 0.008)
        noise = np.random.randn(len(y)) * noise_level
        augmented.append(y + noise)
    
    # 2. Volume variation (simulates different mic distances)
    if random.random() > 0.3:
        gain = random.uniform(0.6, 1.4)
        augmented.append(np.clip(y * gain, -1.0, 1.0))
    
    # 3. Simulate low-quality microphone (high-pass + low-pass filter)
    if random.random() > 0.5:
        try:
            from scipy import signal
            # High-pass filter at 100Hz (removes low rumble)
            b, a = signal.butter(2, 100 / (sr / 2), btype='high')
            y_filtered = signal.filtfilt(b, a, y)
            # Low-pass filter at 7000Hz (simulates limited bandwidth mic)
            b, a = signal.butter(2, 7000 / (sr / 2), btype='low')
            y_filtered = signal.filtfilt(b, a, y_filtered)
            augmented.append(y_filtered.astype(np.float32))
        except:
            pass
    
    # 4. Add slight echo/reverb (simulates room acoustics)
    if random.random() > 0.6:
        delay_samples = int(sr * random.uniform(0.02, 0.05))  # 20-50ms delay
        decay = random.uniform(0.1, 0.3)
        y_echo = np.zeros(len(y))
        y_echo[:len(y)-delay_samples] = y[delay_samples:] * decay
        augmented.append(np.clip(y + y_echo, -1.0, 1.0).astype(np.float32))
    
    # 5. Subtle time stretch (simulates speech rate variation)
    if random.random() > 0.7:
        stretch_factor = random.uniform(0.95, 1.05)
        try:
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
            if len(y_stretched) > len(y):
                y_stretched = y_stretched[:len(y)]
            else:
                y_stretched = np.pad(y_stretched, (0, len(y) - len(y_stretched)))
            augmented.append(y_stretched)
        except:
            pass
    
    # Extra augmentations for underrepresented classes (Spanish)
    if extra_augmentation:
        # 6. Pitch shift
        if random.random() > 0.4:
            try:
                n_steps = random.uniform(-1, 1)
                y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                augmented.append(y_pitch)
            except:
                pass
        
        # 7. More aggressive noise
        if random.random() > 0.5:
            noise_level = random.uniform(0.005, 0.015)
            noise = np.random.randn(len(y)) * noise_level
            augmented.append(np.clip(y + noise, -1.0, 1.0))
        
        # 8. Different volume level
        if random.random() > 0.4:
            gain = random.uniform(0.5, 0.8)
            augmented.append(np.clip(y * gain, -1.0, 1.0))
    
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


def extract_features(audio_path: str, augment: bool = True, extra_augment: bool = False) -> list:
    """
    Extract MFCC features from audio file with optional augmentation
    
    Args:
        audio_path: path to audio file
        augment: whether to apply augmentation
        extra_augment: whether to apply extra augmentation (for Spanish)
    
    Returns:
        List of numpy arrays of shape (time_frames, 120)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        if augment:
            # Get augmented versions
            audio_versions = augment_audio(y, sr, extra_augmentation=extra_augment)
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
        # Apply extra augmentation for Spanish (0) and German (3) to improve detection
        extra_aug = (label == 0) or (label == 3)  # Spanish and German get extra augmentation
        for audio_file in tqdm(audio_files, desc=f"   Processing"):
            features_list = extract_features(str(audio_file), augment=True, extra_augment=extra_aug)
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
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    
    # Fine-tune weights based on testing:
    # - Spanish (0) needs boost for better detection
    # - French (2) needs slight boost
    # - German (3) now balanced (normalization fixed the over-prediction issue)
    class_weights_arr[0] *= 1.8  # Spanish - boost
    class_weights_arr[2] *= 1.2  # French - slight boost
    class_weights_arr[3] *= 1.3  # German - boost to improve detection
    
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
            patience=10,  # More patience for better convergence
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
