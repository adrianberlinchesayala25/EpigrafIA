"""
Test model with MP3 files from each language
"""
import requests
import os

# Test files from each language
test_files = {
    'Español': r'data\Common Voice\Audios Español\clips',
    'Inglés': r'data\Common Voice\Audios Ingles\clips',
    'Francés': r'data\Common Voice\Audios Frances\clips',
    'Alemán': r'data\Common Voice\Audios Aleman\clips'
}

LABELS = ['Español', 'Inglés', 'Francés', 'Alemán']

print('='*60)
print('🧪 TESTING MODEL WITH MP3 FILES')
print('='*60)

correct = 0
total = 0

for expected_lang, path in test_files.items():
    files = [f for f in os.listdir(path) if f.endswith('.mp3')][:3]
    print(f'\n📂 Testing {expected_lang}:')
    
    for f in files:
        filepath = os.path.join(path, f)
        with open(filepath, 'rb') as audio_file:
            try:
                response = requests.post(
                    'http://localhost:8000/api/analyze',
                    files={'audio': (f, audio_file, 'audio/mpeg')},
                    timeout=30
                )
            except Exception as e:
                print(f'   ❌ Connection error: {e}')
                continue
        
        if response.ok:
            result = response.json()
            pred = LABELS[result['language_prediction']]
            conf = result['language_confidence'] * 100
            status = '✅' if pred == expected_lang else '❌'
            print(f'   {status} {f[:30]}: {pred} ({conf:.1f}%)')
            total += 1
            if pred == expected_lang:
                correct += 1
        else:
            print(f'   ❌ Error: {response.status_code}')

print(f'\n📊 Results: {correct}/{total} correct ({correct/total*100:.1f}%)')
