"""Test model predictions with dataset files"""
import sys
sys.path.insert(0, '.')
from backend.predict import AudioPredictor

# Load predictor with the model path
MODEL_PATH = 'outputs/models_trained/language_model_best.keras'
predictor = AudioPredictor(language_model_path=MODEL_PATH)
LABELS = ['Español', 'Inglés', 'Francés', 'Alemán']

# Test files - multiple samples per language
files = [
    # Español - varios archivos
    ('data/Common Voice/Audios Español/clips/common_voice_es_18306544.mp3', 'Español'),
    ('data/Common Voice/Audios Español/clips/common_voice_es_18306545.mp3', 'Español'),
    ('data/Common Voice/Audios Español/clips/common_voice_es_18306546.mp3', 'Español'),
    # Inglés
    ('data/Common Voice/Audios Ingles/clips/common_voice_en_1.mp3', 'Inglés'),
    ('data/Common Voice/Audios Ingles/clips/common_voice_en_10.mp3', 'Inglés'),
    # Francés
    ('data/Common Voice/Audios Frances/clips/common_voice_fr_17305733.mp3', 'Francés'),
    ('data/Common Voice/Audios Frances/clips/common_voice_fr_17305735.mp3', 'Francés'),
    # Alemán
    ('data/Common Voice/Audios Aleman/clips/common_voice_de_17298952.mp3', 'Alemán'),
    ('data/Common Voice/Audios Aleman/clips/common_voice_de_17298953.mp3', 'Alemán'),
]

print('\n=== TEST CON ARCHIVOS MP3 DEL DATASET ===\n')
correct = 0
for filepath, expected in files:
    with open(filepath, 'rb') as f:
        audio_bytes = f.read()
    result = predictor.predict_language(audio_bytes)
    pred = LABELS[result['prediction']]
    conf = result['confidence'] * 100
    probs = result['probabilities']
    status = '✅' if pred == expected else '❌'
    if pred == expected:
        correct += 1
    print(f'{status} {expected}: Predicción={pred} ({conf:.1f}%)')
    print(f'   Probabilidades: ES={probs[0]*100:.1f}% EN={probs[1]*100:.1f}% FR={probs[2]*100:.1f}% DE={probs[3]*100:.1f}%')

print(f'\nPrecisión: {correct}/{len(files)} ({correct/len(files)*100:.0f}%)')
