"""
🎵 Audio Preprocessing for Language Detection
==============================================
Feature extraction and augmentation for language classification.
"""

import numpy as np
import librosa
from typing import Optional, Tuple, List, Union
from pathlib import Path
import random

from .config import LanguageConfig, DEFAULT_CONFIG


class AudioAugmenter:
    """
    Advanced audio augmentation for language detection.
    
    Implements:
    - Time-domain augmentations (noise, volume, pitch, stretch)
    - Frequency-domain augmentations (SpecAugment)
    - Mixing augmentations (Mixup, CutMix)
    """
    
    def __init__(self, config: LanguageConfig = None):
        self.config = config or DEFAULT_CONFIG
    
    # ============================================
    # Time-Domain Augmentations
    # ============================================
    
    def add_noise(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Add background noise to simulate different environments"""
        if random.random() > self.config.noise_prob:
            return y
        
        min_level, max_level = self.config.noise_level_range
        noise_level = random.uniform(min_level, max_level)
        noise = np.random.randn(len(y)) * noise_level
        
        return np.clip(y + noise, -1.0, 1.0).astype(np.float32)
    
    def adjust_volume(self, y: np.ndarray) -> np.ndarray:
        """Adjust volume to simulate different microphone distances"""
        if random.random() > self.config.volume_prob:
            return y
        
        min_gain, max_gain = self.config.volume_range
        gain = random.uniform(min_gain, max_gain)
        
        return np.clip(y * gain, -1.0, 1.0).astype(np.float32)
    
    def pitch_shift(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Shift pitch to simulate voice variation"""
        if random.random() > self.config.pitch_shift_prob:
            return y
        
        min_steps, max_steps = self.config.pitch_shift_range
        n_steps = random.uniform(min_steps, max_steps)
        
        try:
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            return y_shifted.astype(np.float32)
        except Exception:
            return y
    
    def time_stretch(self, y: np.ndarray, target_len: int = None) -> np.ndarray:
        """Stretch/compress time to simulate speech rate variation"""
        if random.random() > self.config.time_stretch_prob:
            return y
        
        min_rate, max_rate = self.config.time_stretch_range
        rate = random.uniform(min_rate, max_rate)
        
        try:
            y_stretched = librosa.effects.time_stretch(y, rate=rate)
            
            # Ensure same length
            if target_len is None:
                target_len = len(y)
            
            if len(y_stretched) > target_len:
                y_stretched = y_stretched[:target_len]
            elif len(y_stretched) < target_len:
                y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)))
            
            return y_stretched.astype(np.float32)
        except Exception:
            return y
    
    def add_reverb(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Add room reverb/echo effect"""
        if random.random() > self.config.reverb_prob:
            return y
        
        # Simple echo simulation
        delay_samples = int(sr * random.uniform(0.02, 0.05))  # 20-50ms
        decay = random.uniform(0.1, 0.3)
        
        y_echo = np.zeros(len(y))
        if len(y) > delay_samples:
            y_echo[:len(y)-delay_samples] = y[delay_samples:] * decay
        
        return np.clip(y + y_echo, -1.0, 1.0).astype(np.float32)
    
    def simulate_microphone(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Simulate low-quality microphone (bandpass filter)"""
        if random.random() > 0.3:
            return y
        
        try:
            from scipy import signal
            
            # High-pass filter at 100Hz
            b, a = signal.butter(2, 100 / (sr / 2), btype='high')
            y_filtered = signal.filtfilt(b, a, y)
            
            # Low-pass filter at 7000Hz
            b, a = signal.butter(2, 7000 / (sr / 2), btype='low')
            y_filtered = signal.filtfilt(b, a, y_filtered)
            
            return y_filtered.astype(np.float32)
        except Exception:
            return y
    
    def augment_audio(self, y: np.ndarray, sr: int = 16000, 
                      strong: bool = False) -> np.ndarray:
        """Apply all time-domain augmentations"""
        target_len = len(y)
        
        # Apply augmentations
        y = self.add_noise(y, sr)
        y = self.adjust_volume(y)
        
        if strong:
            y = self.pitch_shift(y, sr)
            y = self.time_stretch(y, target_len)
            y = self.add_reverb(y, sr)
            y = self.simulate_microphone(y, sr)
        
        return y
    
    # ============================================
    # Frequency-Domain Augmentations (SpecAugment)
    # ============================================
    
    def freq_mask(self, features: np.ndarray) -> np.ndarray:
        """Apply frequency masking (SpecAugment)"""
        if not self.config.use_specaugment:
            return features
        
        features = features.copy()
        n_freq = features.shape[1]
        
        for _ in range(self.config.num_freq_masks):
            f = random.randint(0, self.config.freq_mask_param)
            f0 = random.randint(0, max(0, n_freq - f))
            features[:, f0:f0+f] = 0
        
        return features
    
    def time_mask(self, features: np.ndarray) -> np.ndarray:
        """Apply time masking (SpecAugment)"""
        if not self.config.use_specaugment:
            return features
        
        features = features.copy()
        n_time = features.shape[0]
        
        for _ in range(self.config.num_time_masks):
            t = random.randint(0, self.config.time_mask_param)
            t0 = random.randint(0, max(0, n_time - t))
            features[t0:t0+t, :] = 0
        
        return features
    
    def spec_augment(self, features: np.ndarray) -> np.ndarray:
        """Apply full SpecAugment (freq + time masking)"""
        features = self.freq_mask(features)
        features = self.time_mask(features)
        return features
    
    # ============================================
    # Mixing Augmentations
    # ============================================
    
    @staticmethod
    def mixup(x1: np.ndarray, y1: int, x2: np.ndarray, y2: int,
              alpha: float = 0.4, num_classes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation: blend two samples.
        
        Returns:
            Mixed features and soft labels
        """
        lam = np.random.beta(alpha, alpha)
        
        x_mixed = lam * x1 + (1 - lam) * x2
        
        # Create soft labels
        y_mixed = np.zeros(num_classes)
        y_mixed[y1] = lam
        y_mixed[y2] = 1 - lam
        
        return x_mixed, y_mixed
    
    @staticmethod
    def cutmix(x1: np.ndarray, y1: int, x2: np.ndarray, y2: int,
               alpha: float = 1.0, num_classes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutMix augmentation: cut and paste regions.
        
        Returns:
            Mixed features and soft labels
        """
        lam = np.random.beta(alpha, alpha)
        
        # Get dimensions
        time_steps, features = x1.shape
        
        # Calculate cut region
        cut_ratio = np.sqrt(1 - lam)
        cut_len = int(time_steps * cut_ratio)
        
        # Random position
        cx = np.random.randint(time_steps)
        x_start = np.clip(cx - cut_len // 2, 0, time_steps)
        x_end = np.clip(cx + cut_len // 2, 0, time_steps)
        
        # Apply cutmix
        x_mixed = x1.copy()
        x_mixed[x_start:x_end, :] = x2[x_start:x_end, :]
        
        # Adjust lambda based on actual cut
        lam = 1 - (x_end - x_start) / time_steps
        
        # Create soft labels
        y_mixed = np.zeros(num_classes)
        y_mixed[y1] = lam
        y_mixed[y2] = 1 - lam
        
        return x_mixed, y_mixed


class LanguagePreprocessor:
    """
    Audio preprocessing and feature extraction for language detection.
    
    Extracts MFCC features with delta and delta-delta coefficients,
    normalized and ready for CNN input.
    """
    
    def __init__(self, config: LanguageConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.augmenter = AudioAugmenter(self.config)
    
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file and resample if needed"""
        audio_path = str(audio_path)
        
        y, sr = librosa.load(
            audio_path,
            sr=self.config.sample_rate,
            mono=True
        )
        
        return y, sr
    
    def pad_or_trim(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """Ensure audio has correct duration"""
        if sr is None:
            sr = self.config.sample_rate
        
        target_samples = int(sr * self.config.duration)
        
        if len(y) < target_samples:
            # Pad with zeros
            y = np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            # Trim
            y = y[:target_samples]
        
        return y
    
    def extract_mfcc(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract MFCC features with deltas.
        
        Returns:
            Features of shape (time_frames, n_mfcc * 3)
        """
        if sr is None:
            sr = self.config.sample_rate
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        # Compute delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Stack: (n_mfcc * 3, time_frames)
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        # Transpose to (time_frames, features)
        features = features.T
        
        return features
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features (zero mean, unit variance)"""
        mean = features.mean()
        std = features.std() + 1e-8
        
        normalized = (features - mean) / std
        
        return normalized.astype(np.float32)
    
    def extract_features(self, audio_path: Union[str, Path],
                        augment: bool = False,
                        strong_augment: bool = False) -> Optional[np.ndarray]:
        """
        Extract features from audio file.
        
        Args:
            audio_path: Path to audio file
            augment: Apply data augmentation
            strong_augment: Apply stronger augmentation
            
        Returns:
            Features of shape (time_frames, n_features) or None if error
        """
        try:
            # Load audio
            y, sr = self.load_audio(audio_path)
            
            # Apply time-domain augmentation
            if augment:
                y = self.augmenter.augment_audio(y, sr, strong=strong_augment)
            
            # Pad/trim to target duration
            y = self.pad_or_trim(y, sr)
            
            # Extract MFCC features
            features = self.extract_mfcc(y, sr)
            
            # Normalize
            features = self.normalize_features(features)
            
            # Apply SpecAugment if enabled
            if augment and self.config.use_specaugment:
                features = self.augmenter.spec_augment(features)
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_features_from_array(self, y: np.ndarray, sr: int = None,
                                    augment: bool = False) -> Optional[np.ndarray]:
        """Extract features from audio array directly"""
        try:
            if sr is None:
                sr = self.config.sample_rate
            
            # Apply augmentation
            if augment:
                y = self.augmenter.augment_audio(y, sr)
            
            # Pad/trim
            y = self.pad_or_trim(y, sr)
            
            # Extract features
            features = self.extract_mfcc(y, sr)
            
            # Normalize
            features = self.normalize_features(features)
            
            # SpecAugment
            if augment and self.config.use_specaugment:
                features = self.augmenter.spec_augment(features)
            
            return features
            
        except Exception as e:
            print(f"Error processing audio array: {e}")
            return None
    
    def extract_multiple_augmented(self, audio_path: Union[str, Path],
                                   num_augmentations: int = 3) -> List[np.ndarray]:
        """
        Extract multiple augmented versions of the same audio.
        
        Returns:
            List of feature arrays
        """
        features_list = []
        
        # Original (no augmentation)
        features = self.extract_features(audio_path, augment=False)
        if features is not None:
            features_list.append(features)
        
        # Augmented versions
        for i in range(num_augmentations):
            # Alternate between normal and strong augmentation
            strong = (i % 2 == 0)
            features = self.extract_features(audio_path, augment=True, 
                                            strong_augment=strong)
            if features is not None:
                features_list.append(features)
        
        return features_list


def test_preprocessing():
    """Test preprocessing pipeline"""
    import matplotlib.pyplot as plt
    
    config = LanguageConfig()
    preprocessor = LanguagePreprocessor(config)
    augmenter = AudioAugmenter(config)
    
    print("\n" + "="*60)
    print("🎵 PREPROCESSING TEST")
    print("="*60)
    
    # Generate test signal
    sr = config.sample_rate
    duration = config.duration
    t = np.linspace(0, duration, int(sr * duration))
    
    # Synthetic speech-like signal
    signal = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)
    signal = signal.astype(np.float32)
    
    print(f"\n📊 Input signal shape: {signal.shape}")
    print(f"   Duration: {duration}s at {sr}Hz")
    
    # Extract features
    features = preprocessor.extract_features_from_array(signal, sr, augment=False)
    print(f"\n✅ Extracted features shape: {features.shape}")
    print(f"   Expected: {config.input_shape}")
    
    # Test augmentation
    features_aug = preprocessor.extract_features_from_array(signal, sr, augment=True)
    print(f"\n🔧 Augmented features shape: {features_aug.shape}")
    
    # Test SpecAugment
    features_spec = augmenter.spec_augment(features.copy())
    print(f"📊 SpecAugment applied successfully")
    
    # Test Mixup
    features2 = preprocessor.extract_features_from_array(signal * 0.8, sr)
    x_mix, y_mix = augmenter.mixup(features, 0, features2, 1, alpha=0.4)
    print(f"🔀 Mixup: shape={x_mix.shape}, labels={y_mix}")
    
    print("\n✅ All preprocessing tests passed!")
    
    return features


if __name__ == "__main__":
    test_preprocessing()
