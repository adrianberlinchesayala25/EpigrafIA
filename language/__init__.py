"""
🗣️ EpigrafIA - Language Detection Module
==========================================
State-of-the-art language detection from audio.

Features:
- Advanced CNN architecture with SE attention and residual connections
- SpecAugment, Mixup, CutMix data augmentation
- Label smoothing, focal loss, cosine annealing
- Test-Time Augmentation (TTA)
- K-Fold cross validation with ensemble

Supported languages:
- Español (Spanish)
- Inglés (English)
- Francés (French)
- Alemán (German)
"""

from .config import LanguageConfig, LANGUAGE_LABELS, LANGUAGE_CODES
from .preprocessing import LanguagePreprocessor, AudioAugmenter
from .model import create_language_model, create_advanced_language_model
from .train import LanguageTrainer
from .evaluate import LanguageEvaluator
from .predict import LanguagePredictor

__all__ = [
    'LanguageConfig',
    'LANGUAGE_LABELS',
    'LANGUAGE_CODES',
    'LanguagePreprocessor',
    'AudioAugmenter',
    'create_language_model',
    'create_advanced_language_model',
    'LanguageTrainer',
    'LanguageEvaluator',
    'LanguagePredictor',
]
