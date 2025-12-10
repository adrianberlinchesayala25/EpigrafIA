"""
Test para diagnosticar diferencias entre grabación en vivo y MP3 del dataset
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import librosa
from pathlib import Path

# Configuración igual que el modelo
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

def extract_and_analyze(filepath, label):
    """Extrae features y analiza características del audio"""
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    
    # Stats básicos
    rms = np.sqrt(np.mean(y**2))
    max_amp = np.abs(y).max()
    duration = len(y) / sr
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zero_crossing = librosa.feature.zero_crossing_rate(y).mean()
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_mean = mfccs.mean(axis=1)
    mfcc_std = mfccs.std(axis=1)
    
    print(f"\n{'='*60}")
    print(f"📁 {label}")
    print(f"{'='*60}")
    print(f"Duración: {duration:.2f}s | Samples: {len(y)}")
    print(f"RMS: {rms:.6f} | Max: {max_amp:.6f}")
    print(f"Spectral Centroid: {spectral_centroid:.2f} Hz")
    print(f"Spectral Bandwidth: {spectral_bandwidth:.2f} Hz")
    print(f"Spectral Rolloff: {spectral_rolloff:.2f} Hz")
    print(f"Zero Crossing Rate: {zero_crossing:.4f}")
    print(f"MFCC[0] mean: {mfcc_mean[0]:.2f} (energy)")
    print(f"MFCC[1-5] means: {mfcc_mean[1:6]}")
    
    return {
        'rms': rms,
        'max': max_amp,
        'centroid': spectral_centroid,
        'bandwidth': spectral_bandwidth,
        'rolloff': spectral_rolloff,
        'zcr': zero_crossing,
        'mfcc_mean': mfcc_mean,
        'mfcc_std': mfcc_std
    }

# Analizar archivos del dataset (MP3)
print("\n🎵 ANÁLISIS DE ARCHIVOS MP3 DEL DATASET")
mp3_files = [
    ('data/Common Voice/Audios Español/clips/common_voice_es_18306544.mp3', 'Español MP3 #1'),
    ('data/Common Voice/Audios Ingles/clips/common_voice_en_1.mp3', 'Inglés MP3 #1'),
]

mp3_stats = []
for filepath, label in mp3_files:
    if Path(filepath).exists():
        stats = extract_and_analyze(filepath, label)
        mp3_stats.append(stats)

# Mostrar comparación
print("\n" + "="*60)
print("📊 COMPARACIÓN MP3 vs WAV ESPERADO")
print("="*60)
print("""
DIFERENCIAS TÍPICAS ENTRE MP3 Y WAV:

1. MP3 tiene rolloff más bajo (~4000-8000 Hz por codificación)
2. MP3 puede tener artefactos en altas frecuencias
3. WAV de micrófono tiene espectro más amplio
4. WAV de micrófono puede tener más ruido de fondo
5. El bitrate de MP3 afecta las características espectrales

SOLUCIÓN PROPUESTA:
- Agregar audios WAV al entrenamiento
- O aplicar filtro lowpass para simular MP3
- O normalizar características espectrales
""")

# Verificar si hay archivos WAV de prueba
wav_test = Path('test_recordings')
if wav_test.exists():
    wav_files = list(wav_test.glob('*.wav'))
    if wav_files:
        print("\n🎤 ANÁLISIS DE GRABACIONES WAV")
        for f in wav_files[:3]:
            extract_and_analyze(str(f), f"WAV: {f.name}")
