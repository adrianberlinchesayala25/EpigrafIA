"""
🔮 Módulo de Inferencia para Detección de Spoofing
==================================================

Este módulo proporciona predicción en producción para detectar
si un audio es humano real o generado por IA.

Uso:
    predictor = SpoofingPredictor(model_path="model.keras")
    result = predictor.predict_file("audio.wav")
    # {'is_ai': True, 'probability': 0.87, 'label': 'spoof'}
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .preprocessing import AudioPreprocessor
from .config import AUDIO_CONFIG, LABELS

logger = logging.getLogger(__name__)


class SpoofingPredictor:
    """
    Predictor de spoofing para uso en producción.
    
    Proporciona una interfaz simple para detectar si un audio
    es humano real (bonafide) o generado por IA (spoof).
    
    Attributes:
        model: Modelo Keras cargado
        preprocessor: Preprocesador de audio
        threshold: Umbral de decisión
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model: Optional[keras.Model] = None,
        threshold: float = 0.5,
        preprocessor: Optional[AudioPreprocessor] = None
    ):
        """
        Inicializa el predictor.
        
        Args:
            model_path: Ruta al modelo guardado (.keras o .h5)
            model: Modelo Keras ya cargado (alternativa a model_path)
            threshold: Umbral de decisión (default 0.5)
            preprocessor: Preprocesador personalizado
        """
        # Cargar modelo
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Debe proporcionar model o model_path")
        
        self.threshold = threshold
        
        # Inicializar preprocesador
        self.preprocessor = preprocessor or AudioPreprocessor(
            sample_rate=AUDIO_CONFIG.sample_rate,
            n_mfcc=AUDIO_CONFIG.n_mfcc,
            n_fft=AUDIO_CONFIG.n_fft,
            hop_length=AUDIO_CONFIG.hop_length,
            n_mels=AUDIO_CONFIG.n_mels,
            fixed_length=AUDIO_CONFIG.fixed_length
        )
        
        logger.info(f"✅ SpoofingPredictor inicializado (threshold={threshold})")
    
    def _load_model(self, model_path: Union[str, Path]) -> keras.Model:
        """Carga el modelo desde archivo"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        logger.info(f"📥 Cargando modelo desde: {model_path}")
        
        # Suprimir warnings de TF durante la carga
        tf.get_logger().setLevel('ERROR')
        
        model = keras.models.load_model(str(model_path))
        
        return model
    
    def predict_file(
        self,
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Predice si un archivo de audio es humano o IA.
        
        Args:
            file_path: Ruta al archivo de audio
        
        Returns:
            Diccionario con:
            - is_ai: bool - True si es IA, False si es humano
            - probability: float - Probabilidad de ser IA (0-1)
            - label: str - 'spoof' o 'human'
            - confidence: float - Confianza de la predicción (0-1)
        """
        # Preprocesar audio
        features = self.preprocessor.process_file(file_path)
        
        # Añadir dimensión de batch
        features = np.expand_dims(features, axis=0)
        
        # Predecir
        probability = float(self.model.predict(features, verbose=0)[0, 0])
        
        # Decisión binaria
        is_ai = probability >= self.threshold
        label = 'spoof' if is_ai else 'human'
        
        # Confianza (qué tan seguro está el modelo)
        confidence = probability if is_ai else (1 - probability)
        
        return {
            'is_ai': is_ai,
            'probability': probability,
            'label': label,
            'confidence': confidence,
            'threshold': self.threshold
        }
    
    def predict_bytes(
        self,
        audio_bytes: bytes,
        file_format: str = "wav"
    ) -> Dict[str, Any]:
        """
        Predice desde bytes de audio (para APIs).
        
        Args:
            audio_bytes: Bytes del archivo de audio
            file_format: Formato del audio (wav, mp3, etc.)
        
        Returns:
            Diccionario con predicción
        """
        # Preprocesar desde bytes
        features = self.preprocessor.process_bytes(audio_bytes, file_format)
        
        # Añadir dimensión de batch
        features = np.expand_dims(features, axis=0)
        
        # Predecir
        probability = float(self.model.predict(features, verbose=0)[0, 0])
        
        is_ai = probability >= self.threshold
        label = 'spoof' if is_ai else 'human'
        confidence = probability if is_ai else (1 - probability)
        
        return {
            'is_ai': is_ai,
            'probability': probability,
            'label': label,
            'confidence': confidence,
            'threshold': self.threshold
        }
    
    def predict_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Predice desde array de numpy.
        
        Args:
            audio_array: Array de audio (float32, mono)
            sample_rate: Sample rate del audio
        
        Returns:
            Diccionario con predicción
        """
        # Normalizar
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val
        
        # Resamplear si es necesario
        if sample_rate != self.preprocessor.sample_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=self.preprocessor.sample_rate
            )
        
        # Extraer features
        mfccs = self.preprocessor.extract_mfcc(audio_array)
        mfccs = self.preprocessor.pad_or_truncate(mfccs)
        
        # Añadir dimensión de batch
        features = np.expand_dims(mfccs, axis=0)
        
        # Predecir
        probability = float(self.model.predict(features, verbose=0)[0, 0])
        
        is_ai = probability >= self.threshold
        label = 'spoof' if is_ai else 'human'
        confidence = probability if is_ai else (1 - probability)
        
        return {
            'is_ai': is_ai,
            'probability': probability,
            'label': label,
            'confidence': confidence,
            'threshold': self.threshold
        }
    
    def predict_batch(
        self,
        file_paths: list
    ) -> list:
        """
        Predice múltiples archivos en batch.
        
        Args:
            file_paths: Lista de rutas a archivos
        
        Returns:
            Lista de diccionarios con predicciones
        """
        # Preprocesar todos
        features_list = []
        valid_indices = []
        
        for i, path in enumerate(file_paths):
            try:
                feat = self.preprocessor.process_file(path)
                features_list.append(feat)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Error procesando {path}: {e}")
        
        if not features_list:
            return []
        
        # Batch prediction
        features = np.array(features_list)
        probabilities = self.model.predict(features, verbose=0).flatten()
        
        # Construir resultados
        results = [None] * len(file_paths)
        
        for idx, prob in zip(valid_indices, probabilities):
            is_ai = prob >= self.threshold
            results[idx] = {
                'is_ai': bool(is_ai),
                'probability': float(prob),
                'label': 'spoof' if is_ai else 'human',
                'confidence': float(prob if is_ai else (1 - prob)),
                'file': str(file_paths[idx])
            }
        
        return results
    
    def set_threshold(self, threshold: float):
        """Actualiza el umbral de decisión"""
        if not 0 < threshold < 1:
            raise ValueError("Threshold debe estar entre 0 y 1")
        self.threshold = threshold
        logger.info(f"🔧 Threshold actualizado a: {threshold}")


# ============================================
# Singleton para uso global (opcional)
# ============================================

_global_predictor: Optional[SpoofingPredictor] = None


def get_predictor(
    model_path: Optional[str] = None,
    threshold: float = 0.5
) -> SpoofingPredictor:
    """
    Obtiene instancia global del predictor (singleton pattern).
    
    Útil para APIs donde queremos reutilizar el modelo cargado.
    
    Args:
        model_path: Ruta al modelo (solo necesario la primera vez)
        threshold: Umbral de decisión
    
    Returns:
        Instancia de SpoofingPredictor
    """
    global _global_predictor
    
    if _global_predictor is None:
        if model_path is None:
            raise ValueError("model_path requerido para inicialización")
        _global_predictor = SpoofingPredictor(
            model_path=model_path,
            threshold=threshold
        )
    
    return _global_predictor


def predict_audio(
    audio_input: Union[str, bytes, np.ndarray],
    model_path: Optional[str] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Función de conveniencia para predicción rápida.
    
    Args:
        audio_input: Ruta, bytes o array de audio
        model_path: Ruta al modelo
        threshold: Umbral de decisión
    
    Returns:
        Diccionario con predicción
    """
    predictor = get_predictor(model_path, threshold)
    
    if isinstance(audio_input, str) or isinstance(audio_input, Path):
        return predictor.predict_file(audio_input)
    elif isinstance(audio_input, bytes):
        return predictor.predict_bytes(audio_input)
    elif isinstance(audio_input, np.ndarray):
        return predictor.predict_array(audio_input)
    else:
        raise ValueError(f"Tipo de entrada no soportado: {type(audio_input)}")


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("🔮 MÓDULO DE PREDICCIÓN DE SPOOFING")
    print("="*70)
    print("""
USO BÁSICO:
-----------
    from spoofing.predict import SpoofingPredictor
    
    # Inicializar con modelo
    predictor = SpoofingPredictor(model_path="model.keras")
    
    # Predecir archivo
    result = predictor.predict_file("audio.wav")
    print(result)
    # {'is_ai': True, 'probability': 0.87, 'label': 'spoof', 'confidence': 0.87}

PARA API:
---------
    # Desde bytes (FastAPI/Flask)
    result = predictor.predict_bytes(audio_bytes, file_format="wav")
    
    # Batch processing
    results = predictor.predict_batch(["audio1.wav", "audio2.wav"])

SINGLETON (recomendado para APIs):
----------------------------------
    from spoofing.predict import get_predictor
    
    # Primera llamada inicializa
    predictor = get_predictor(model_path="model.keras")
    
    # Llamadas posteriores reutilizan
    predictor = get_predictor()
    result = predictor.predict_file("audio.wav")
""")
