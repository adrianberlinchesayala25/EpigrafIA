"""
⚙️ Language Detection Configuration
====================================
Central configuration for all language detection parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


# ============================================
# Language Labels
# ============================================

LANGUAGE_LABELS = {
    0: 'Español',
    1: 'Inglés',
    2: 'Francés',
    3: 'Alemán'
}

LANGUAGE_CODES = {
    'es': 0,
    'en': 1,
    'fr': 2,
    'de': 3
}

LANGUAGE_FOLDERS = {
    'Audios Español': 0,
    'Audios Ingles': 1,
    'Audios Frances': 2,
    'Audios Aleman': 3
}


@dataclass
class LanguageConfig:
    """Configuration for language detection training and inference"""
    
    # ============================================
    # Paths
    # ============================================
    data_dir: Path = field(default_factory=lambda: Path("data/Common Voice"))
    output_dir: Path = field(default_factory=lambda: Path("outputs/language_models"))
    log_dir: Path = field(default_factory=lambda: Path("outputs/language_logs"))
    
    # ============================================
    # Audio Processing (MUST match inference)
    # ============================================
    sample_rate: int = 16000
    duration: float = 3.0  # seconds
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    
    # Computed feature dimensions
    # time_frames = (sample_rate * duration) / hop_length ≈ 94
    # features = n_mfcc * 3 (mfcc + delta + delta2) = 120
    
    # ============================================
    # Data Loading
    # ============================================
    samples_per_language: int = 2000  # More samples for better generalization
    validation_split: float = 0.15
    test_split: float = 0.10
    random_seed: int = 42
    
    # ============================================
    # Model Architecture
    # ============================================
    num_classes: int = 4
    model_type: str = "advanced"  # "basic", "advanced", "efficientnet"
    
    # Advanced model settings
    base_filters: int = 64
    num_blocks: int = 4
    use_se_attention: bool = True
    se_ratio: int = 16
    use_residual: bool = True
    
    # Dropout rates
    dropout_rate: float = 0.4
    spatial_dropout: float = 0.1
    
    # ============================================
    # Training Hyperparameters
    # ============================================
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    
    # Learning rate schedule
    lr_schedule: str = "cosine"  # "cosine", "reduce_on_plateau", "warmup_cosine"
    warmup_epochs: int = 5
    
    # ============================================
    # Regularization
    # ============================================
    label_smoothing: float = 0.1
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    
    # Class balancing
    use_class_weights: bool = True
    class_weight_boost: Dict[int, float] = field(default_factory=lambda: {
        0: 1.5,  # Español - boost
        1: 1.0,  # Inglés - baseline
        2: 1.2,  # Francés - slight boost
        3: 1.3,  # Alemán - boost
    })
    
    # ============================================
    # Data Augmentation
    # ============================================
    use_augmentation: bool = True
    
    # SpecAugment
    use_specaugment: bool = True
    freq_mask_param: int = 10  # F in SpecAugment paper
    time_mask_param: int = 20  # T in SpecAugment paper
    num_freq_masks: int = 2
    num_time_masks: int = 2
    
    # Mixup / CutMix
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    
    # Audio augmentation
    noise_prob: float = 0.5
    noise_level_range: tuple = (0.001, 0.01)
    volume_prob: float = 0.5
    volume_range: tuple = (0.6, 1.4)
    pitch_shift_prob: float = 0.3
    pitch_shift_range: tuple = (-2, 2)  # semitones
    time_stretch_prob: float = 0.3
    time_stretch_range: tuple = (0.9, 1.1)
    reverb_prob: float = 0.3
    
    # ============================================
    # Test-Time Augmentation (TTA)
    # ============================================
    use_tta: bool = True
    tta_augmentations: int = 5
    
    # ============================================
    # Cross Validation
    # ============================================
    use_kfold: bool = False
    n_folds: int = 5
    
    # ============================================
    # Callbacks
    # ============================================
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    
    # ============================================
    # Hardware
    # ============================================
    use_mixed_precision: bool = True
    num_workers: int = 4
    prefetch_buffer: int = 2
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def input_shape(self) -> tuple:
        """Calculate input shape based on audio parameters"""
        # Time frames from audio duration
        num_samples = int(self.sample_rate * self.duration)
        time_frames = num_samples // self.hop_length + 1  # ≈ 94
        # Features: MFCC + delta + delta2
        features = self.n_mfcc * 3  # 120
        return (time_frames, features)
    
    @property
    def expected_time_frames(self) -> int:
        """Expected number of time frames"""
        num_samples = int(self.sample_rate * self.duration)
        return num_samples // self.hop_length + 1
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'n_mfcc': self.n_mfcc,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'num_classes': self.num_classes,
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'label_smoothing': self.label_smoothing,
            'use_focal_loss': self.use_focal_loss,
            'use_mixup': self.use_mixup,
            'use_specaugment': self.use_specaugment,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LanguageConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# Default configuration
DEFAULT_CONFIG = LanguageConfig()


if __name__ == "__main__":
    config = LanguageConfig()
    print("\n" + "="*60)
    print("⚙️  LANGUAGE DETECTION CONFIG")
    print("="*60)
    
    print(f"\n📁 Paths:")
    print(f"   Data: {config.data_dir}")
    print(f"   Output: {config.output_dir}")
    
    print(f"\n🎵 Audio:")
    print(f"   Sample rate: {config.sample_rate}")
    print(f"   Duration: {config.duration}s")
    print(f"   MFCCs: {config.n_mfcc}")
    
    print(f"\n🧠 Model:")
    print(f"   Type: {config.model_type}")
    print(f"   Input shape: {config.input_shape}")
    print(f"   Classes: {config.num_classes}")
    
    print(f"\n🏋️ Training:")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   LR: {config.learning_rate}")
    print(f"   Label smoothing: {config.label_smoothing}")
    
    print(f"\n🔧 Augmentation:")
    print(f"   SpecAugment: {config.use_specaugment}")
    print(f"   Mixup: {config.use_mixup}")
    print(f"   CutMix: {config.use_cutmix}")
