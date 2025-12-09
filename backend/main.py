"""
🎤 EpigrafIA Backend - FastAPI Server
=====================================
API para detección de idioma y acento usando Deep Learning

Endpoints:
- POST /api/analyze - Analiza audio y devuelve predicciones
- GET /api/health - Estado del servidor
- GET /api/models/status - Estado de los modelos cargados
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from predict import AudioPredictor

# ============================================
# Configuration
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "outputs" / "models_trained"

# Labels for predictions
LANGUAGE_LABELS = ['Español', 'Inglés', 'Francés', 'Alemán']
ACCENT_LABELS = [
    'España', 'México', 'UK', 'USA',
    'Francia', 'Quebec', 'Alemania', 'Austria'
]

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="EpigrafIA API",
    description="API de detección de idioma y acento con Deep Learning",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[AudioPredictor] = None


# ============================================
# Startup / Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global predictor
    
    logger.info("🚀 Starting EpigrafIA API...")
    
    try:
        predictor = AudioPredictor(
            language_model_path=MODELS_DIR / "language_model.keras",
            accent_model_path=MODELS_DIR / "accent_model.keras"
        )
        logger.info("✅ Models loaded successfully!")
        
    except FileNotFoundError as e:
        logger.warning(f"⚠️ Models not found: {e}")
        logger.warning("   The API will start but predictions won't work.")
        logger.warning("   Train the models first using the notebooks.")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global predictor
    if predictor:
        predictor.cleanup()
    logger.info("👋 EpigrafIA API shutting down...")


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "EpigrafIA API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /api/analyze",
            "health": "GET /api/health",
            "models": "GET /api/models/status",
            "docs": "GET /api/docs"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": predictor is not None and predictor.models_loaded
    }


@app.get("/api/models/status")
async def models_status():
    """Get status of loaded models"""
    if predictor is None:
        return {
            "loaded": False,
            "error": "Predictor not initialized"
        }
    
    return {
        "loaded": predictor.models_loaded,
        "language_model": predictor.language_model is not None,
        "accent_model": predictor.accent_model is not None,
        "language_labels": LANGUAGE_LABELS,
        "accent_labels": ACCENT_LABELS
    }


@app.post("/api/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    """
    Analyze audio file and return language/accent predictions
    
    Accepts: WAV, MP3, WebM, OGG audio files
    Returns: JSON with predictions and probabilities
    """
    # Validate file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check content type
    allowed_types = ['audio/wav', 'audio/webm', 'audio/mp3', 'audio/mpeg', 
                     'audio/ogg', 'audio/x-wav', 'audio/wave']
    
    content_type = audio.content_type or ''
    if not any(t in content_type for t in ['audio', 'webm', 'wav', 'mp3', 'ogg']):
        logger.warning(f"Unexpected content type: {content_type}")
    
    # Check if models are loaded
    if predictor is None or not predictor.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train the models first."
        )
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"📥 Received audio: {audio.filename} ({len(audio_data)} bytes)")
        
        # Run prediction
        result = predictor.predict(audio_data)
        
        # Format response
        language_probs = result['language_probabilities']
        accent_probs = result['accent_probabilities']
        
        response = {
            "success": True,
            "language": {
                "detected": LANGUAGE_LABELS[language_probs.argmax()],
                "confidence": float(language_probs.max()),
                "probabilities": {
                    label: float(prob) 
                    for label, prob in zip(LANGUAGE_LABELS, language_probs)
                }
            },
            "accent": {
                "detected": ACCENT_LABELS[accent_probs.argmax()],
                "confidence": float(accent_probs.max()),
                "probabilities": {
                    label: float(prob) 
                    for label, prob in zip(ACCENT_LABELS, accent_probs)
                }
            }
        }
        
        logger.info(f"✅ Prediction: {response['language']['detected']} - {response['accent']['detected']}")
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
