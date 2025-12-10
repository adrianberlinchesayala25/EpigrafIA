"""Test audio characteristics"""
import librosa
import numpy as np
from pathlib import Path

# Load a training sample
train_files = list(Path('data/Common Voice/Audios Español').glob('*.mp3'))
if not train_files:
    train_files = list(Path('data/Common Voice/Audios Espanol').glob('*.mp3'))

if train_files:
    train_file = train_files[0]
    y, sr = librosa.load(str(train_file), sr=16000)
    print('=== TRAINING AUDIO (MP3) ===')
    print(f'File: {train_file.name}')
    print(f'Duration: {len(y)/sr:.2f}s')
    print(f'Max amplitude: {np.abs(y).max():.4f}')
    print(f'RMS: {np.sqrt(np.mean(y**2)):.4f}')
else:
    print('No training files found')
