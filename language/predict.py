"""
🔮 Language Detection Predictor
================================
Production-ready inference with TTA support.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
import json

import tensorflow as tf
from tensorflow import keras

from .config import LanguageConfig, LANGUAGE_LABELS
from .preprocessing import LanguagePreprocessor, AudioAugmenter


class LanguagePredictor:
    """
    Language predictor for production inference.
    
    Features:
    - Load from saved model
    - Single file and batch prediction
    - Test-Time Augmentation (TTA)
    - Confidence thresholding
    - Ensemble prediction (multiple models)
    """
    
    def __init__(self, 
                 model_path: Union[str, Path] = None,
                 config: LanguageConfig = None,
                 use_tta: bool = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model
            config: Configuration
            use_tta: Whether to use TTA (default from config)
        """
        self.config = config or LanguageConfig()
        
        if use_tta is not None:
            self.use_tta = use_tta
        else:
            self.use_tta = self.config.use_tta
        
        self.preprocessor = LanguagePreprocessor(self.config)
        self.augmenter = AudioAugmenter(self.config)
        
        # Load model
        if model_path is None:
            model_path = self.config.output_dir / "language_model_final.keras"
        
        self.model = self._load_model(model_path)
        self.model_path = model_path
        
        print(f"✅ Language predictor initialized")
        print(f"   Model: {model_path}")
        print(f"   TTA: {'Enabled' if self.use_tta else 'Disabled'}")
    
    def _load_model(self, model_path: Union[str, Path]) -> keras.Model:
        """Load model from path"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Custom objects for focal loss if used
        custom_objects = {
            'FocalLoss': keras.losses.Loss,  # Placeholder, won't be used for inference
        }
        
        try:
            model = keras.models.load_model(str(model_path), compile=False)
        except Exception:
            model = keras.models.load_model(str(model_path), 
                                            custom_objects=custom_objects,
                                            compile=False)
        
        return model
    
    def predict_file(self, audio_path: Union[str, Path],
                     use_tta: bool = None,
                     return_all_probs: bool = False) -> Dict:
        """
        Predict language from audio file.
        
        Args:
            audio_path: Path to audio file
            use_tta: Override TTA setting
            return_all_probs: Return probabilities for all classes
            
        Returns:
            Dictionary with prediction results
        """
        if use_tta is None:
            use_tta = self.use_tta
        
        # Extract features
        if use_tta:
            # Multiple augmented versions
            features_list = self.preprocessor.extract_multiple_augmented(
                audio_path,
                num_augmentations=self.config.tta_augmentations - 1
            )
            
            if not features_list:
                return {'error': 'Could not extract features', 'success': False}
            
            X = np.array(features_list)
            
            # Predict all versions and average
            predictions = self.model.predict(X, verbose=0)
            avg_prediction = predictions.mean(axis=0)
            
        else:
            features = self.preprocessor.extract_features(audio_path, augment=False)
            
            if features is None:
                return {'error': 'Could not extract features', 'success': False}
            
            X = np.expand_dims(features, axis=0)
            avg_prediction = self.model.predict(X, verbose=0)[0]
        
        # Get prediction
        predicted_class = int(np.argmax(avg_prediction))
        confidence = float(avg_prediction[predicted_class])
        
        result = {
            'success': True,
            'language': LANGUAGE_LABELS[predicted_class],
            'language_code': predicted_class,
            'confidence': confidence,
            'confidence_percent': f"{confidence*100:.1f}%",
        }
        
        if return_all_probs:
            result['probabilities'] = {
                LANGUAGE_LABELS[i]: float(avg_prediction[i])
                for i in range(len(avg_prediction))
            }
        
        return result
    
    def predict_array(self, y: np.ndarray, sr: int = None,
                      use_tta: bool = None) -> Dict:
        """
        Predict language from audio array.
        
        Args:
            y: Audio signal array
            sr: Sample rate (default from config)
            use_tta: Override TTA setting
            
        Returns:
            Prediction result dictionary
        """
        if sr is None:
            sr = self.config.sample_rate
        
        if use_tta is None:
            use_tta = self.use_tta
        
        # Extract features
        if use_tta:
            features_list = []
            
            # Original
            features = self.preprocessor.extract_features_from_array(y, sr, augment=False)
            if features is not None:
                features_list.append(features)
            
            # Augmented versions
            for _ in range(self.config.tta_augmentations - 1):
                features = self.preprocessor.extract_features_from_array(y, sr, augment=True)
                if features is not None:
                    features_list.append(features)
            
            if not features_list:
                return {'error': 'Could not extract features', 'success': False}
            
            X = np.array(features_list)
            predictions = self.model.predict(X, verbose=0)
            avg_prediction = predictions.mean(axis=0)
            
        else:
            features = self.preprocessor.extract_features_from_array(y, sr, augment=False)
            
            if features is None:
                return {'error': 'Could not extract features', 'success': False}
            
            X = np.expand_dims(features, axis=0)
            avg_prediction = self.model.predict(X, verbose=0)[0]
        
        predicted_class = int(np.argmax(avg_prediction))
        confidence = float(avg_prediction[predicted_class])
        
        return {
            'success': True,
            'language': LANGUAGE_LABELS[predicted_class],
            'language_code': predicted_class,
            'confidence': confidence,
            'confidence_percent': f"{confidence*100:.1f}%",
            'probabilities': {
                LANGUAGE_LABELS[i]: float(avg_prediction[i])
                for i in range(len(avg_prediction))
            }
        }
    
    def predict_batch(self, audio_paths: List[Union[str, Path]],
                      use_tta: bool = None) -> List[Dict]:
        """
        Predict languages for multiple files.
        
        Args:
            audio_paths: List of audio file paths
            use_tta: Override TTA setting
            
        Returns:
            List of prediction results
        """
        results = []
        
        for path in audio_paths:
            result = self.predict_file(path, use_tta=use_tta)
            result['file'] = str(path)
            results.append(result)
        
        return results
    
    def predict_features(self, X: np.ndarray, 
                         use_tta: bool = None) -> np.ndarray:
        """
        Predict from pre-extracted features.
        
        Args:
            X: Features of shape (samples, time, features)
            use_tta: Use TTA on features (SpecAugment only)
            
        Returns:
            Predictions of shape (samples, num_classes)
        """
        if use_tta is None:
            use_tta = self.use_tta
        
        if use_tta:
            all_preds = []
            
            # Original
            pred = self.model.predict(X, verbose=0)
            all_preds.append(pred)
            
            # Augmented
            for _ in range(self.config.tta_augmentations - 1):
                X_aug = np.array([self.augmenter.spec_augment(x.copy()) for x in X])
                pred = self.model.predict(X_aug, verbose=0)
                all_preds.append(pred)
            
            return np.mean(all_preds, axis=0)
        else:
            return self.model.predict(X, verbose=0)
    
    def get_top_languages(self, audio_path: Union[str, Path],
                          top_k: int = 3) -> List[Dict]:
        """
        Get top-k language predictions.
        
        Args:
            audio_path: Path to audio file
            top_k: Number of top predictions
            
        Returns:
            List of top predictions with confidence
        """
        result = self.predict_file(audio_path, return_all_probs=True)
        
        if not result['success']:
            return [result]
        
        probs = result['probabilities']
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'language': lang, 'confidence': conf, 'confidence_percent': f"{conf*100:.1f}%"}
            for lang, conf in sorted_probs[:top_k]
        ]


class EnsemblePredictor:
    """
    Ensemble predictor using multiple models.
    
    Combines predictions from K-Fold trained models
    or different architectures for more robust results.
    """
    
    def __init__(self, model_paths: List[Union[str, Path]],
                 config: LanguageConfig = None,
                 weights: List[float] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            model_paths: List of model paths
            config: Configuration
            weights: Optional weights for each model
        """
        self.config = config or LanguageConfig()
        self.preprocessor = LanguagePreprocessor(self.config)
        
        # Load all models
        self.models = []
        for path in model_paths:
            try:
                model = keras.models.load_model(str(path), compile=False)
                self.models.append(model)
                print(f"   Loaded: {path}")
            except Exception as e:
                print(f"   ⚠️ Could not load {path}: {e}")
        
        if not self.models:
            raise ValueError("No models loaded successfully")
        
        # Set weights
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            assert len(weights) == len(self.models)
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        print(f"\n✅ Ensemble predictor initialized with {len(self.models)} models")
    
    def predict(self, audio_path: Union[str, Path]) -> Dict:
        """
        Ensemble prediction.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Combined prediction result
        """
        features = self.preprocessor.extract_features(audio_path, augment=False)
        
        if features is None:
            return {'error': 'Could not extract features', 'success': False}
        
        X = np.expand_dims(features, axis=0)
        
        # Get predictions from all models
        ensemble_pred = np.zeros(self.config.num_classes)
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X, verbose=0)[0]
            ensemble_pred += weight * pred
        
        predicted_class = int(np.argmax(ensemble_pred))
        confidence = float(ensemble_pred[predicted_class])
        
        return {
            'success': True,
            'language': LANGUAGE_LABELS[predicted_class],
            'language_code': predicted_class,
            'confidence': confidence,
            'confidence_percent': f"{confidence*100:.1f}%",
            'num_models': len(self.models),
            'probabilities': {
                LANGUAGE_LABELS[i]: float(ensemble_pred[i])
                for i in range(len(ensemble_pred))
            }
        }


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔮 LANGUAGE PREDICTOR TEST")
    print("="*60)
    
    config = LanguageConfig()
    
    model_path = config.output_dir / "language_model_final.keras"
    
    if model_path.exists():
        predictor = LanguagePredictor(model_path, config)
        
        # Test with synthetic audio
        import librosa
        sr = config.sample_rate
        duration = config.duration
        
        # Generate test signal
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
        
        result = predictor.predict_array(test_audio, sr)
        
        print(f"\n🎤 Test prediction:")
        print(f"   Language: {result['language']}")
        print(f"   Confidence: {result['confidence_percent']}")
        
    else:
        print(f"\n⚠️ Model not found at {model_path}")
        print("   Train a model first using: python run_language.py train")
