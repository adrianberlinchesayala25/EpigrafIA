"""
⚙️ Configuración del Sistema de Detección de Spoofing
======================================================
Parámetros centralizados para preprocesado, modelo y entrenamiento
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AudioConfig:
    """Configuración de procesamiento de audio"""
    
    # Parámetros de audio
    sample_rate: int = 16000          # Hz - estándar para voz
    duration: float = 4.0             # segundos máximos por audio
    
    # Parámetros MFCC
    n_mfcc: int = 40                  # Coeficientes MFCC
    n_fft: int = 512                  # Ventana FFT (32ms @ 16kHz)
    hop_length: int = 160             # 10ms hop (16000 * 0.01)
    n_mels: int = 80                  # Bandas mel para cálculo
    fmin: int = 20                    # Frecuencia mínima Hz
    fmax: int = 8000                  # Frecuencia máxima Hz
    
    # Longitud fija de frames
    # Con 4s de audio, hop=160, sr=16000: 4*16000/160 = 400 frames
    fixed_length: int = 400
    
    @property
    def input_shape(self) -> Tuple[int, int]:
        """Shape de entrada para el modelo: (time_steps, features)"""
        return (self.fixed_length, self.n_mfcc)


@dataclass
class ModelConfig:
    """Configuración de la arquitectura del modelo"""
    
    # Arquitectura CNN
    conv_filters: Tuple[int, ...] = (64, 128, 256, 512)
    conv_kernels: Tuple[int, ...] = (5, 5, 3, 3)
    dropout_rates: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.4)
    
    # Capas densas
    dense_units: Tuple[int, ...] = (256, 128)
    dense_dropout: float = 0.5
    
    # Activación
    leaky_alpha: float = 0.1


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    
    # Hiperparámetros
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # ReduceLROnPlateau
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-7
    
    # Reproducibilidad
    random_seed: int = 42


@dataclass
class PathConfig:
    """Rutas del proyecto"""
    
    base_dir: Path = Path(".")
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data" / "spoofing"
    
    @property
    def human_dir(self) -> Path:
        return self.data_dir / "humano"
    
    @property
    def spoof_dir(self) -> Path:
        return self.data_dir / "spoof"
    
    @property
    def metadata_path(self) -> Path:
        return self.base_dir / "data" / "metadata.csv"
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "outputs" / "spoofing_models"
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "outputs" / "spoofing_logs"


# ============================================
# Instancias por defecto
# ============================================

AUDIO_CONFIG = AudioConfig()
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
PATH_CONFIG = PathConfig()


# ============================================
# Labels
# ============================================

LABELS = {
    0: "human",      # bonafide
    1: "spoof"       # IA generada
}

LABEL_NAMES = {
    "bonafide": 0,
    "human": 0,
    "spoof": 1,
    "fake": 1,
    "ai": 1
}
