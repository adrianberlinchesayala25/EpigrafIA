"""
Analizar características de audio por idioma
"""
import librosa
import numpy as np
import os

# Analizar RMS de archivos de cada idioma
data_paths = {
    'Español': r'data\Common Voice\Audios Español\clips',
    'Inglés': r'data\Common Voice\Audios Ingles\clips',
    'Francés': r'data\Common Voice\Audios Frances\clips',
    'Alemán': r'data\Common Voice\Audios Aleman\clips'
}

print('=' * 60)
print('📊 ANÁLISIS DE VOLUMEN (RMS) POR IDIOMA')
print('=' * 60)

for lang, path in data_paths.items():
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith('.mp3')][:30]  # 30 samples
        rms_values = []
        max_values = []
        for f in files:
            try:
                y, sr = librosa.load(os.path.join(path, f), sr=16000, mono=True)
                rms = np.sqrt(np.mean(y**2))
                max_amp = np.abs(y).max()
                rms_values.append(rms)
                max_values.append(max_amp)
            except:
                pass
        if rms_values:
            print(f'\n🎤 {lang}:')
            print(f'   RMS promedio: {np.mean(rms_values):.4f}')
            print(f'   RMS min/max: {np.min(rms_values):.4f} / {np.max(rms_values):.4f}')
            print(f'   Amplitud max promedio: {np.mean(max_values):.4f}')
    else:
        print(f'\n❌ {lang}: Ruta no encontrada')
