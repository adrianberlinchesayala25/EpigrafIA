"""
ðŸŽ¤ EpigrafIA - Hugging Face Spaces API
======================================
Backend con FastAPI + TensorFlow para detecciÃ³n de idioma/acento
Desplegable en Hugging Face Spaces (Docker SDK)

Endpoints:
- POST /api/predict - Analiza audio y devuelve predicciones
- GET /api/health - Estado del servidor
"""

import os
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ============================================
# Configuration
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Audio processing config
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

# Labels
LANGUAGE_LABELS = ['EspaÃ±ol', 'InglÃ©s', 'FrancÃ©s', 'AlemÃ¡n']
ACCENT_LABELS = ['EspaÃ±a', 'MÃ©xico', 'UK', 'USA', 'Francia', 'Quebec', 'Alemania', 'Austria']

# Global model
language_model = None
accent_model = None

# ============================================
# Model Loading
# ============================================

def load_models():
    """Load TensorFlow models"""
    global language_model, accent_model
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Try different model paths
        model_paths = [
            Path("outputs/models_trained/language_model_best.keras"),
            Path("outputs/models_trained/language_model.keras"),
            Path("models/language_model_best.keras"),
            Path("models/language_model.keras"),
            Path("language_model_best.keras"),
            Path("language_model.keras"),
        ]
        
        for path in model_paths:
            if path.exists():
                logger.info(f"ðŸ“¥ Loading language model from {path}")
                language_model = tf.keras.models.load_model(str(path))
                logger.info(f"âœ… Model loaded! Input: {language_model.input_shape}, Output: {language_model.output_shape}")
                break
        
        if language_model is None:
            logger.warning("âš ï¸ No language model found. Predictions will fail.")
            
    except Exception as e:
        logger.error(f"âŒ Error loading models: {e}")

# ============================================
# Audio Processing
# ============================================

def extract_features(audio_data: bytes) -> np.ndarray:
    """Extract MFCC features from audio bytes"""
    import librosa
    
    # Detect format and save to temp file
    is_webm = audio_data[:4] == b'\x1a\x45\xdf\xa3'
    suffix = '.webm' if is_webm else '.wav'
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    
    wav_path = None
    try:
        load_path = tmp_path
        
        # Convert WebM to WAV if needed
        if is_webm:
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(tmp_path, format="webm")
                audio_segment = audio_segment.set_channels(1).set_frame_rate(SAMPLE_RATE)
                wav_path = tmp_path.replace('.webm', '.wav')
                audio_segment.export(wav_path, format="wav")
                load_path = wav_path
            except Exception as e:
                logger.warning(f"WebM conversion failed: {e}")
        
        # Load audio
        y, sr = librosa.load(load_path, sr=SAMPLE_RATE, mono=True)
        
        # Ensure correct duration
        min_samples = SAMPLE_RATE * DURATION
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)), mode='constant')
        else:
            y = y[:min_samples]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Stack and normalize
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        features = (features - features.mean()) / (features.std() + 1e-8)
        features = features.T
        features = np.expand_dims(features, axis=0)
        
        return features
        
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
        if wav_path:
            try:
                os.unlink(wav_path)
            except:
                pass

# ============================================
# FastAPI App
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    load_models()
    yield
    logger.info("ðŸ‘‹ Shutting down...")

app = FastAPI(
    title="EpigrafIA API",
    description="API de detecciÃ³n de idioma y acento con Deep Learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - permitir frontend de Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "EpigrafIA API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": language_model is not None
    }

@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": language_model is not None
    }

@app.post("/api/predict")
async def predict(audio: UploadFile = File(...)):
    """
    Analyze audio and return language/accent predictions
    """
    if language_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import tensorflow as tf
        
        # Read audio
        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"ðŸ“¥ Received: {audio.filename} ({len(audio_data)} bytes)")
        
        # Extract features and predict
        features = extract_features(audio_data)
        language_probs = language_model.predict(features, verbose=0)[0]
        
        # Ensure probabilities sum to 1
        if not np.isclose(language_probs.sum(), 1.0, atol=0.01):
            language_probs = tf.nn.softmax(language_probs).numpy()
        
        lang_idx = int(np.argmax(language_probs))
        
        response = {
            "success": True,
            "language": {
                "detected": LANGUAGE_LABELS[lang_idx],
                "confidence": float(language_probs.max()),
                "probabilities": {
                    label: float(prob) 
                    for label, prob in zip(LANGUAGE_LABELS, language_probs)
                }
            },
            "language_prediction": lang_idx,
            "language_confidence": float(language_probs.max()),
            "accent": {
                "detected": "No disponible",
                "confidence": 0.0,
                "probabilities": {}
            }
        }
        
        logger.info(f"âœ… Prediction: {LANGUAGE_LABELS[lang_idx]} ({language_probs.max()*100:.1f}%)")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    """Alias for /api/predict"""
    return await predict(audio)

# ============================================
# Run
# ============================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)

