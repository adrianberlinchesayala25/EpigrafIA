"""
🛡️ EpigrafIA Spoofing Detection Module
=======================================
Sistema de detección de audio generado por IA vs humano real

Componentes:
- config: Configuración centralizada
- preprocessing: Extracción de MFCCs y preprocesado
- model: Arquitectura CNN 1D para detección de spoofing
- train: Pipeline de entrenamiento
- evaluate: Métricas y evaluación
- predict: Inferencia en producción
- api: Endpoints FastAPI

Uso rápido:
    from spoofing import SpoofingPredictor
    
    predictor = SpoofingPredictor(model_path="model.keras")
    result = predictor.predict_file("audio.wav")
    print(f"Es IA: {result['is_spoof']} ({result['confidence']*100:.1f}%)")
"""

__version__ = "1.0.0"
__author__ = "Adrian Berlinches"

# Exports principales
from .config import AUDIO_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG
from .predict import SpoofingPredictor
from .train import SpoofingTrainer, train_spoofing_model
from .evaluate import SpoofingEvaluator
from .model import create_spoofing_detector, compile_model

__all__ = [
    # Config
    "AUDIO_CONFIG",
    "MODEL_CONFIG", 
    "TRAINING_CONFIG",
    "PATH_CONFIG",
    # Clases principales
    "SpoofingPredictor",
    "SpoofingTrainer",
    "SpoofingEvaluator",
    # Funciones
    "train_spoofing_model",
    "create_spoofing_detector",
    "compile_model",
]
