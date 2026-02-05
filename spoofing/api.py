"""
🚀 API FastAPI para Detección de Spoofing de Audio
==================================================

Endpoint de producción para detectar si un audio es humano real
o generado por inteligencia artificial.

Endpoints:
- POST /api/spoofing/detect - Analiza audio y devuelve predicción
- GET /api/spoofing/health - Estado del servicio
- GET /api/spoofing/info - Información del modelo

Uso:
    uvicorn spoofing.api:app --host 0.0.0.0 --port 8000
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Imports del módulo
from .predict import SpoofingPredictor
from .config import AUDIO_CONFIG

# ============================================
# Configuración
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ruta al modelo (configurable por variable de entorno)
MODEL_PATH = os.environ.get(
    'SPOOFING_MODEL_PATH',
    'outputs/spoofing_models/spoofing_best.keras'
)

# Predictor global
predictor: Optional[SpoofingPredictor] = None


# ============================================
# Lifecycle
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación"""
    global predictor
    
    # Startup: Cargar modelo
    logger.info("🚀 Iniciando API de Spoofing Detection...")
    
    try:
        predictor = SpoofingPredictor(model_path=MODEL_PATH)
        logger.info(f"✅ Modelo cargado desde: {MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(f"⚠️ Modelo no encontrado: {MODEL_PATH}")
        logger.warning("   La API iniciará pero /detect no funcionará")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Cerrando API de Spoofing Detection")


# ============================================
# App FastAPI
# ============================================

app = FastAPI(
    title="EpigrafIA Spoofing Detection API",
    description="""
    API para detectar si un audio es humano real (bonafide) o 
    generado por inteligencia artificial (spoof/deepfake).
    
    ## Características
    - Detección en tiempo real
    - Soporta múltiples formatos (WAV, MP3, FLAC)
    - Threshold configurable
    - Respuesta con probabilidad y confianza
    
    ## Uso
    1. Envía un archivo de audio a `/api/spoofing/detect`
    2. Recibe respuesta con `is_ai`, `probability` y `confidence`
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Modelos Pydantic
# ============================================

class SpoofingResponse(BaseModel):
    """Respuesta de detección de spoofing"""
    is_ai: bool = Field(..., description="True si el audio es generado por IA")
    probability: float = Field(..., ge=0, le=1, description="Probabilidad de ser IA (0-1)")
    label: str = Field(..., description="'spoof' o 'human'")
    confidence: float = Field(..., ge=0, le=1, description="Confianza de la predicción")
    threshold: float = Field(..., description="Umbral de decisión utilizado")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_ai": True,
                "probability": 0.87,
                "label": "spoof",
                "confidence": 0.87,
                "threshold": 0.5
            }
        }


class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str
    model_loaded: bool
    model_path: str


class InfoResponse(BaseModel):
    """Información del modelo"""
    model_name: str
    input_shape: list
    threshold: float
    supported_formats: list
    audio_config: dict


# ============================================
# Endpoints
# ============================================

@app.get("/api/spoofing/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Verifica el estado del servicio.
    
    Returns:
        Estado del servicio y si el modelo está cargado
    """
    return HealthResponse(
        status="healthy" if predictor else "degraded",
        model_loaded=predictor is not None,
        model_path=MODEL_PATH
    )


@app.get("/api/spoofing/info", response_model=InfoResponse, tags=["Info"])
async def get_info():
    """
    Obtiene información del modelo y configuración.
    
    Returns:
        Detalles del modelo cargado
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return InfoResponse(
        model_name="SpoofingDetectorCNN",
        input_shape=list(AUDIO_CONFIG.input_shape),
        threshold=predictor.threshold,
        supported_formats=["wav", "mp3", "flac", "ogg", "m4a"],
        audio_config={
            "sample_rate": AUDIO_CONFIG.sample_rate,
            "n_mfcc": AUDIO_CONFIG.n_mfcc,
            "fixed_length": AUDIO_CONFIG.fixed_length,
            "duration_seconds": AUDIO_CONFIG.duration
        }
    )


@app.post("/api/spoofing/detect", response_model=SpoofingResponse, tags=["Detection"])
async def detect_spoofing(
    audio: UploadFile = File(..., description="Archivo de audio a analizar"),
    threshold: float = Query(0.5, ge=0.1, le=0.9, description="Umbral de decisión")
):
    """
    Detecta si un audio es humano real o generado por IA.
    
    ## Parámetros
    - **audio**: Archivo de audio (WAV, MP3, FLAC, etc.)
    - **threshold**: Umbral de decisión (default 0.5)
      - Menor threshold = más sensible a detectar IA (más falsos positivos)
      - Mayor threshold = más conservador (menos falsos positivos)
    
    ## Respuesta
    - **is_ai**: True si se detecta como IA, False si es humano
    - **probability**: Probabilidad de ser IA (0 = seguro humano, 1 = seguro IA)
    - **confidence**: Qué tan seguro está el modelo de su decisión
    - **label**: 'spoof' o 'human'
    
    ## Interpretación
    - probability < 0.3: Muy probablemente humano
    - probability 0.3-0.7: Zona de incertidumbre
    - probability > 0.7: Muy probablemente IA
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Verifique que el modelo esté cargado."
        )
    
    # Validar tipo de archivo
    valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'}
    file_ext = Path(audio.filename).suffix.lower() if audio.filename else '.wav'
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: {file_ext}. Use: {valid_extensions}"
        )
    
    try:
        # Leer bytes del audio
        audio_bytes = await audio.read()
        
        if len(audio_bytes) < 1000:
            raise HTTPException(
                status_code=400,
                detail="Archivo de audio muy pequeño o vacío"
            )
        
        # Actualizar threshold si es diferente
        if threshold != predictor.threshold:
            predictor.set_threshold(threshold)
        
        # Detectar formato
        file_format = file_ext.replace('.', '') if file_ext else 'wav'
        
        # Predecir
        result = predictor.predict_bytes(audio_bytes, file_format)
        
        logger.info(
            f"🎤 Predicción: {result['label']} "
            f"(prob={result['probability']:.3f}, conf={result['confidence']:.3f})"
        )
        
        return SpoofingResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando audio: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando audio: {str(e)}"
        )


@app.post("/api/spoofing/batch", tags=["Detection"])
async def detect_batch(
    audios: list[UploadFile] = File(..., description="Lista de archivos de audio"),
    threshold: float = Query(0.5, ge=0.1, le=0.9)
):
    """
    Detecta spoofing en múltiples archivos.
    
    ## Parámetros
    - **audios**: Lista de archivos de audio
    - **threshold**: Umbral de decisión
    
    ## Respuesta
    Lista de resultados, uno por archivo.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    import tempfile
    
    results = []
    
    for audio in audios:
        try:
            audio_bytes = await audio.read()
            file_ext = Path(audio.filename).suffix.lower().replace('.', '') or 'wav'
            
            if threshold != predictor.threshold:
                predictor.set_threshold(threshold)
            
            result = predictor.predict_bytes(audio_bytes, file_ext)
            result['filename'] = audio.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': audio.filename,
                'error': str(e)
            })
    
    return results


# ============================================
# Error handlers
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor"}
    )


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("🚀 INICIANDO API DE DETECCIÓN DE SPOOFING")
    print("="*70)
    print(f"Modelo: {MODEL_PATH}")
    print("Endpoints:")
    print("  POST /api/spoofing/detect - Detectar spoofing en audio")
    print("  GET  /api/spoofing/health - Estado del servicio")
    print("  GET  /api/spoofing/info   - Información del modelo")
    print("="*70)
    
    uvicorn.run(
        "spoofing.api:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
