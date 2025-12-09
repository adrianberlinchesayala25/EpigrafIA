"""
🎵 Audio Processing Module
==========================
Audio preprocessing and feature extraction for EpigrafIA

Features:
- MFCC extraction (40 coefficients + deltas)
- Audio normalization and trimming
- Batch processing support
- Data augmentation
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Union
import warnings

import numpy as np

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Lazy load librosa (heavy import)
librosa = None
sf = None


def _ensure_librosa():
    """Lazy load librosa"""
    global librosa, sf
    if librosa is None:
        import librosa as lb
        import soundfile as soundfile_module
        librosa = lb
        sf = soundfile_module


# ============================================
# Configuration
# ============================================

class AudioConfig:
    """Audio processing configuration"""
    SAMPLE_RATE: int = 16000
    DURATION: float = 3.0  # seconds
    N_MFCC: int = 40
    N_MELS: int = 128
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    FMIN: int = 20
    FMAX: int = 8000
    
    @classmethod
    def get_n_frames(cls) -> int:
        """Calculate number of time frames for given duration"""
        n_samples = int(cls.SAMPLE_RATE * cls.DURATION)
        return 1 + (n_samples - cls.N_FFT) // cls.HOP_LENGTH


# ============================================
# Feature Extraction
# ============================================

def load_audio(
    file_path: Union[str, Path],
    sr: int = AudioConfig.SAMPLE_RATE,
    duration: Optional[float] = AudioConfig.DURATION,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        duration: Max duration in seconds (None for full audio)
        mono: Convert to mono
    
    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    _ensure_librosa()
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Load with librosa
    y, original_sr = librosa.load(
        str(file_path),
        sr=sr,
        mono=mono,
        duration=duration
    )
    
    return y, sr


def pad_or_trim(
    audio: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Pad audio to target length or trim if too long
    
    Args:
        audio: Audio samples
        target_length: Target number of samples
    
    Returns:
        Padded or trimmed audio
    """
    if len(audio) < target_length:
        # Pad with zeros
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    elif len(audio) > target_length:
        # Trim
        audio = audio[:target_length]
    
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range
    
    Args:
        audio: Audio samples
    
    Returns:
        Normalized audio
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def extract_mfcc(
    audio: np.ndarray,
    sr: int = AudioConfig.SAMPLE_RATE,
    n_mfcc: int = AudioConfig.N_MFCC,
    n_fft: int = AudioConfig.N_FFT,
    hop_length: int = AudioConfig.HOP_LENGTH,
    include_deltas: bool = True
) -> np.ndarray:
    """
    Extract MFCC features from audio
    
    Args:
        audio: Audio samples
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        include_deltas: Include delta and delta² features
    
    Returns:
        MFCC features array (n_features, n_frames)
        If include_deltas=True: n_features = n_mfcc * 3
    """
    _ensure_librosa()
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=AudioConfig.FMIN,
        fmax=AudioConfig.FMAX
    )
    
    if include_deltas:
        # Compute delta and delta² features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Stack features
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    else:
        features = mfccs
    
    return features


def extract_features(
    file_path: Union[str, Path],
    normalize: bool = True
) -> np.ndarray:
    """
    Full feature extraction pipeline for a single audio file
    
    Args:
        file_path: Path to audio file
        normalize: Normalize features to zero mean, unit variance
    
    Returns:
        Feature array shaped (n_frames, n_features) for model input
    """
    # Load audio
    audio, sr = load_audio(file_path)
    
    # Ensure correct length
    target_samples = int(AudioConfig.SAMPLE_RATE * AudioConfig.DURATION)
    audio = pad_or_trim(audio, target_samples)
    
    # Normalize audio
    audio = normalize_audio(audio)
    
    # Extract MFCCs
    features = extract_mfcc(audio, sr)
    
    # Normalize features
    if normalize:
        mean = features.mean()
        std = features.std()
        features = (features - mean) / (std + 1e-8)
    
    # Transpose to (n_frames, n_features) for model input
    features = features.T
    
    return features


# ============================================
# Data Augmentation
# ============================================

def augment_audio(
    audio: np.ndarray,
    sr: int = AudioConfig.SAMPLE_RATE,
    noise_factor: float = 0.005,
    shift_max: float = 0.2,
    pitch_shift_steps: Optional[int] = None,
    time_stretch_rate: Optional[float] = None
) -> np.ndarray:
    """
    Apply data augmentation to audio
    
    Args:
        audio: Audio samples
        sr: Sample rate
        noise_factor: Amount of Gaussian noise to add
        shift_max: Max proportion of audio to shift
        pitch_shift_steps: Semitones to shift pitch (None to skip)
        time_stretch_rate: Time stretch factor (None to skip)
    
    Returns:
        Augmented audio
    """
    _ensure_librosa()
    
    augmented = audio.copy()
    
    # Add noise
    if noise_factor > 0:
        noise = np.random.randn(len(augmented)) * noise_factor
        augmented = augmented + noise
    
    # Time shift
    if shift_max > 0:
        shift_amount = int(len(augmented) * np.random.uniform(-shift_max, shift_max))
        augmented = np.roll(augmented, shift_amount)
    
    # Pitch shift
    if pitch_shift_steps is not None:
        augmented = librosa.effects.pitch_shift(
            augmented, sr=sr, n_steps=pitch_shift_steps
        )
    
    # Time stretch
    if time_stretch_rate is not None:
        augmented = librosa.effects.time_stretch(
            augmented, rate=time_stretch_rate
        )
        # Pad or trim to original length
        augmented = pad_or_trim(augmented, len(audio))
    
    return augmented


# ============================================
# Batch Processing
# ============================================

def process_dataset(
    audio_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    extensions: List[str] = ['.wav', '.mp3', '.flac', '.ogg']
) -> List[Tuple[str, np.ndarray]]:
    """
    Process all audio files in a directory
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Optional directory to save features as .npy
        extensions: List of audio file extensions to process
    
    Returns:
        List of (filename, features) tuples
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Directory not found: {audio_dir}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f'**/*{ext}'))
    
    logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")
    
    for audio_file in audio_files:
        try:
            # Extract features
            features = extract_features(audio_file)
            
            # Save if output dir specified
            if output_dir:
                output_path = output_dir / f"{audio_file.stem}.npy"
                np.save(output_path, features)
            
            results.append((audio_file.name, features))
            
        except Exception as e:
            logger.warning(f"Error processing {audio_file}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(results)} files")
    return results


# ============================================
# Utility Functions
# ============================================

def get_audio_info(file_path: Union[str, Path]) -> dict:
    """
    Get information about an audio file
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Dictionary with audio information
    """
    _ensure_librosa()
    
    file_path = Path(file_path)
    
    # Get duration without loading full file
    duration = librosa.get_duration(path=str(file_path))
    
    # Load small sample for sample rate
    y, sr = librosa.load(str(file_path), duration=0.1)
    
    return {
        'path': str(file_path),
        'filename': file_path.name,
        'duration_seconds': duration,
        'sample_rate': sr,
        'file_size_mb': file_path.stat().st_size / (1024 * 1024)
    }


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sr: int = AudioConfig.SAMPLE_RATE
) -> None:
    """
    Save audio array to file
    
    Args:
        audio: Audio samples
        file_path: Output file path
        sr: Sample rate
    """
    _ensure_librosa()
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    sf.write(str(file_path), audio, sr)
    logger.info(f"Saved audio to {file_path}")


# ============================================
# CLI Test
# ============================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        print(f"\n📊 Processing: {audio_path}")
        print("="*50)
        
        # Get info
        info = get_audio_info(audio_path)
        print(f"Duration: {info['duration_seconds']:.2f}s")
        print(f"Sample rate: {info['sample_rate']} Hz")
        print(f"File size: {info['file_size_mb']:.2f} MB")
        
        # Extract features
        features = extract_features(audio_path)
        print(f"\n✅ Features extracted: {features.shape}")
        print(f"   Time frames: {features.shape[0]}")
        print(f"   Features per frame: {features.shape[1]}")
        
    else:
        print("Usage: python process_audio.py <audio_file>")
        print("\nConfiguration:")
        print(f"  Sample rate: {AudioConfig.SAMPLE_RATE} Hz")
        print(f"  Duration: {AudioConfig.DURATION} s")
        print(f"  MFCC coefficients: {AudioConfig.N_MFCC}")
        print(f"  Expected frames: {AudioConfig.get_n_frames()}")
        print(f"  Total features: {AudioConfig.N_MFCC * 3} (with deltas)")
