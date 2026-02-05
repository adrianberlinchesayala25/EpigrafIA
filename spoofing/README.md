# 🛡️ EpigrafIA - Módulo de Detección de Spoofing de Audio

Sistema de detección de audio deepfake/spoofing basado en Deep Learning. Distingue entre voz humana real (bonafide) y audio generado por IA (TTS, Voice Conversion, etc.).

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Dataset](#-dataset)
- [Entrenamiento](#-entrenamiento)
- [API](#-api)
- [Métricas](#-métricas)
- [Estructura del Código](#-estructura-del-código)

## ✨ Características

- **Detección Binaria**: Humano (bonafide) vs IA (spoof)
- **Agnóstico al Idioma**: No aprende idioma, solo características de autenticidad
- **CNN 1D Optimizada**: Arquitectura específica para detección de artefactos de síntesis
- **AUC como Métrica Principal**: Evaluación robusta para aplicaciones de seguridad
- **API REST**: Endpoint FastAPI listo para producción
- **Threshold Configurable**: Ajuste según requisitos de precisión/recall

## 🧠 Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    SPOOFING DETECTOR CNN                     │
├─────────────────────────────────────────────────────────────┤
│ Input: (400, 40) - 400 frames × 40 MFCCs                    │
├─────────────────────────────────────────────────────────────┤
│ Conv Block 1: Conv1D(64, k=5) → BN → LeakyReLU → Pool → D0.2│
│ Conv Block 2: Conv1D(128, k=5) → BN → LeakyReLU → Pool → D0.3│
│ Conv Block 3: Conv1D(256, k=3) → BN → LeakyReLU → Pool → D0.4│
│ Conv Block 4: Conv1D(512, k=3) → BN → LeakyReLU → D0.4      │
├─────────────────────────────────────────────────────────────┤
│ GlobalAveragePooling1D                                       │
├─────────────────────────────────────────────────────────────┤
│ Dense(256) → BN → LeakyReLU → Dropout(0.5)                  │
│ Dense(128) → BN → LeakyReLU → Dropout(0.5)                  │
├─────────────────────────────────────────────────────────────┤
│ Dense(1, sigmoid) → P(spoof | audio)                         │
└─────────────────────────────────────────────────────────────┘
```

### ¿Por qué esta arquitectura?

1. **Kernels decrecientes (5→5→3→3)**: Contexto amplio inicial, refinamiento posterior
2. **Dropout progresivo (0.2→0.4)**: Preserva información básica, regulariza capas profundas
3. **LeakyReLU (α=0.1)**: Evita "dying neurons" para patrones sutiles
4. **GlobalAveragePooling**: Mejor generalización que Flatten, invariancia temporal
5. **BatchNormalization**: Estabilidad y regularización implícita

## 📦 Instalación

```bash
# Clonar repositorio
git clone https://github.com/adrianberlinchesayala25/EpigrafIA.git
cd EpigrafIA

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales:
- TensorFlow >= 2.15.0
- librosa >= 0.10.2
- scikit-learn >= 1.5.0
- FastAPI (para API)
- uvicorn (para API)

## 🚀 Uso Rápido

### Predicción desde Python

```python
from spoofing.predict import SpoofingPredictor

# Inicializar predictor
predictor = SpoofingPredictor(model_path="outputs/spoofing_models/spoofing_best.keras")

# Predecir archivo
result = predictor.predict_file("audio.wav")
print(result)
# {'is_ai': True, 'probability': 0.87, 'label': 'spoof', 'confidence': 0.87}
```

### Predicción desde bytes (para APIs)

```python
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()

result = predictor.predict_bytes(audio_bytes, file_format="wav")
```

### API REST

```bash
# Iniciar servidor
python -m spoofing.api

# O con uvicorn
uvicorn spoofing.api:app --host 0.0.0.0 --port 8001
```

```bash
# Hacer predicción
curl -X POST "http://localhost:8001/api/spoofing/detect" \
  -F "audio=@audio.wav" \
  -F "threshold=0.5"
```

## 📊 Dataset

### Estructura esperada

```
data/
└── audios/
    ├── human/          # Audios humanos reales (bonafide)
    │   ├── audio_001.wav
    │   ├── audio_002.mp3
    │   └── ...
    └── spoof/          # Audios generados por IA
        ├── spoof_001.wav
        ├── spoof_002.flac
        └── ...
```

### Fuentes recomendadas

**Human (Bonafide):**
- [Common Voice](https://commonvoice.mozilla.org/) - Multi-idioma
- [ASVspoof 2019 LA](https://www.asvspoof.org/) - Subset bonafide
- [LibriSpeech](https://www.openslr.org/12/) - Inglés

**Spoof (IA):**
- [ASVspoof 2019 LA](https://www.asvspoof.org/) - TTS y Voice Conversion
- [ASVspoof 2021](https://www.asvspoof.org/) - Más sistemas modernos
- Generar con: ElevenLabs, XTTS, Tacotron, etc.

### Formatos soportados
- WAV, MP3, FLAC, OGG, M4A

## 🏋️ Entrenamiento

### Desde línea de comandos

```bash
python train_spoofing.py \
  --human-dir data/audios/human \
  --spoof-dir data/audios/spoof \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4
```

### Desde notebook

Abre `train/train_spoofing.ipynb` y ejecuta las celdas.

### Configuración

Edita `spoofing/config.py` para ajustar:
- Parámetros de audio (sample_rate, n_mfcc, etc.)
- Arquitectura del modelo
- Hiperparámetros de entrenamiento

## 🌐 API

### Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/spoofing/detect` | Detectar spoofing en audio |
| GET | `/api/spoofing/health` | Estado del servicio |
| GET | `/api/spoofing/info` | Información del modelo |

### Respuesta de `/detect`

```json
{
  "is_ai": true,
  "probability": 0.87,
  "label": "spoof",
  "confidence": 0.87,
  "threshold": 0.5
}
```

### Interpretación

| probability | Interpretación |
|-------------|----------------|
| < 0.3 | Muy probablemente humano |
| 0.3 - 0.7 | Zona de incertidumbre |
| > 0.7 | Muy probablemente IA |

## 📈 Métricas

### ¿Por qué AUC y no solo Accuracy?

1. **Accuracy puede ser engañosa** con datasets balanceados pero diferentes costos de error
2. **AUC mide capacidad de ranking**, no solo aciertos
3. **Independiente del threshold**: Permite elegir umbral óptimo en producción
4. **Estándar en seguridad**: ASVspoof challenge usa EER (derivado de ROC)

### Métricas reportadas

- **Accuracy**: Proporción de predicciones correctas
- **Precision**: De los marcados como spoof, cuántos lo son
- **Recall**: De los spoofs reales, cuántos detectamos
- **F1-Score**: Media armónica de precision y recall
- **AUC-ROC**: Área bajo la curva ROC
- **EER**: Equal Error Rate (FPR = FNR)

### Evaluación

```python
from spoofing.evaluate import SpoofingEvaluator

evaluator = SpoofingEvaluator(model_path="model.keras")
results = evaluator.full_evaluation(X_test, y_test, output_dir="eval_results")
```

## 📁 Estructura del Código

```
spoofing/
├── __init__.py          # Package init
├── config.py            # Configuración centralizada
├── preprocessing.py     # Extracción de MFCCs
├── model.py             # Arquitectura CNN 1D
├── train.py             # Pipeline de entrenamiento
├── evaluate.py          # Métricas y evaluación
├── predict.py           # Inferencia en producción
└── api.py               # FastAPI endpoints

train/
└── train_spoofing.ipynb # Notebook de entrenamiento

train_spoofing.py        # Script de entrenamiento CLI
```

## 🔬 Detalles Técnicos

### Preprocesamiento

1. **Carga**: librosa a 16kHz mono
2. **Normalización**: Peak normalization
3. **MFCCs**: 40 coeficientes, 512 FFT, 160 hop
4. **Normalización z-score**: Por instancia
5. **Padding/Truncado**: A 400 frames

### Por qué MFCCs para Spoofing

Los MFCCs capturan la envolvente espectral que refleja:
- Características del tracto vocal (difíciles de sintetizar perfectamente)
- Micromodulaciones naturales de la voz
- Transiciones fonéticas
- Ruido glotal y respiración

Los sistemas TTS/VC producen espectros "demasiado limpios" o con artefactos sutiles que los MFCCs capturan.

## 📝 Licencia

MIT License - Ver [LICENSE](../LICENSE)

## 👤 Autor

Adrian Berlinches - TFG 2025/2026
