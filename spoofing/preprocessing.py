"""
🎵 Pipeline de Preprocesamiento de Audio para Detección de Spoofing
====================================================================

Este módulo implementa el preprocesado de audio optimizado para detectar
diferencias sutiles entre voz humana real y audio generado por IA.

Características extraídas:
- MFCCs (40 coeficientes): Capturan la envolvente espectral
- Normalización por instancia: Elimina variaciones de volumen
- Longitud fija: Padding/truncado a 400 frames

Por qué MFCCs para Spoofing:
----------------------------
Los MFCCs capturan características del tracto vocal que son difíciles
de sintetizar perfectamente por sistemas TTS/VC:
1. Micromodulaciones naturales de la voz humana
2. Variabilidad en formantes
3. Características de respiración y ruido glotal
4. Transiciones fonéticas naturales

Los sistemas de IA tienden a producir espectros "demasiado limpios"
o con artefactos sutiles que los MFCCs pueden capturar.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Lazy imports
librosa = None
sf = None


def _load_librosa():
    """Carga lazy de librosa para optimizar tiempo de importación"""
    global librosa
    if librosa is None:
        import librosa as lib
        librosa = lib
    return librosa


def _load_soundfile():
    """Carga lazy de soundfile"""
    global sf
    if sf is None:
        import soundfile as soundfile_lib
        sf = soundfile_lib
    return sf


class AudioPreprocessor:
    """
    Preprocesador de audio para detección de spoofing.
    
    Extrae MFCCs normalizados de archivos de audio y los prepara
    para entrada a una CNN 1D.
    
    Attributes:
        sample_rate: Tasa de muestreo objetivo (Hz)
        n_mfcc: Número de coeficientes MFCC
        n_fft: Tamaño de ventana FFT
        hop_length: Salto entre ventanas
        fixed_length: Longitud fija en frames
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: int = 20,
        fmax: int = 8000,
        fixed_length: int = 400
    ):
        """
        Inicializa el preprocesador.
        
        Args:
            sample_rate: Frecuencia de muestreo objetivo
            n_mfcc: Número de coeficientes MFCC a extraer
            n_fft: Tamaño de la ventana FFT
            hop_length: Salto entre ventanas consecutivas
            n_mels: Número de bandas mel
            fmin: Frecuencia mínima para mel
            fmax: Frecuencia máxima para mel
            fixed_length: Longitud fija de salida en frames
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.fixed_length = fixed_length
        
        # Cargar librosa
        _load_librosa()
    
    def load_audio(
        self,
        file_path: Union[str, Path],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Carga un archivo de audio y lo preprocesa.
        
        Args:
            file_path: Ruta al archivo de audio
            normalize: Si normalizar la amplitud
        
        Returns:
            Array de audio (mono, float32)
        
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el audio está vacío o corrupto
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio no encontrado: {file_path}")
        
        try:
            # Cargar audio con librosa (maneja múltiples formatos)
            y, sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                mono=True
            )
            
            # Verificar que el audio no está vacío
            if len(y) == 0:
                raise ValueError(f"Audio vacío: {file_path}")
            
            # Normalización de amplitud
            if normalize:
                y = self._normalize_audio(y)
            
            return y.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error cargando {file_path}: {e}")
            raise
    
    def _normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """
        Normaliza el audio para tener amplitud consistente.
        
        Usamos normalización peak para mantener la dinámica
        pero asegurar que el máximo sea 1.0.
        
        Args:
            y: Array de audio
        
        Returns:
            Audio normalizado
        """
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
        return y
    
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """
        Extrae coeficientes MFCC del audio.
        
        Los MFCCs son ideales para spoofing porque capturan:
        - Envolvente espectral (características del tracto vocal)
        - Información prosódica
        - Microestructura temporal
        
        Args:
            y: Array de audio normalizado
        
        Returns:
            Array de MFCCs: (n_frames, n_mfcc)
        """
        # Extraer MFCCs
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Transponer para tener (time, features)
        mfccs = mfccs.T
        
        # Normalización por instancia (z-score)
        # Crítico para spoofing: elimina variaciones de grabación
        mfccs = self._normalize_features(mfccs)
        
        return mfccs.astype(np.float32)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalización z-score por instancia.
        
        Cada audio se normaliza independientemente para eliminar
        diferencias de grabación (micrófonos, ambientes, etc.)
        
        Args:
            features: Array de características (time, features)
        
        Returns:
            Características normalizadas
        """
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        
        # Evitar división por cero
        std = np.where(std < 1e-8, 1.0, std)
        
        return (features - mean) / std
    
    def pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """
        Ajusta la longitud de las características a tamaño fijo.
        
        - Si es más corto: padding con ceros (post-padding)
        - Si es más largo: truncado centrado
        
        Args:
            features: Array de características (time, features)
        
        Returns:
            Array de tamaño fijo (fixed_length, features)
        """
        current_length = features.shape[0]
        
        if current_length == self.fixed_length:
            return features
        
        elif current_length < self.fixed_length:
            # Padding con ceros al final
            pad_amount = self.fixed_length - current_length
            padding = np.zeros((pad_amount, features.shape[1]), dtype=np.float32)
            return np.vstack([features, padding])
        
        else:
            # Truncado centrado (mantener la parte central del audio)
            start = (current_length - self.fixed_length) // 2
            return features[start:start + self.fixed_length]
    
    def process_file(
        self,
        file_path: Union[str, Path]
    ) -> np.ndarray:
        """
        Procesa un archivo de audio completo.
        
        Pipeline: Load → Normalize → MFCC → Pad/Truncate
        
        Args:
            file_path: Ruta al archivo de audio
        
        Returns:
            Array de características listo para el modelo: (fixed_length, n_mfcc)
        """
        # 1. Cargar audio
        y = self.load_audio(file_path, normalize=True)
        
        # 2. Extraer MFCCs
        mfccs = self.extract_mfcc(y)
        
        # 3. Ajustar longitud
        mfccs = self.pad_or_truncate(mfccs)
        
        return mfccs
    
    def process_bytes(
        self,
        audio_bytes: bytes,
        file_format: str = "wav"
    ) -> np.ndarray:
        """
        Procesa audio desde bytes (para API).
        
        Args:
            audio_bytes: Bytes del archivo de audio
            file_format: Formato del audio (wav, mp3, etc.)
        
        Returns:
            Array de características listo para el modelo
        """
        import tempfile
        import os
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_format}",
            delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            # Procesar archivo
            features = self.process_file(tmp_path)
            return features
        finally:
            # Limpiar archivo temporal
            os.unlink(tmp_path)


class DatasetLoader:
    """
    Cargador de dataset para entrenamiento de detección de spoofing.
    
    Carga audios de directorios human/ y spoof/ o desde un CSV de metadatos.
    """
    
    def __init__(
        self,
        preprocessor: Optional[AudioPreprocessor] = None,
        human_dir: Optional[Union[str, Path]] = None,
        spoof_dir: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None
    ):
        """
        Inicializa el cargador de dataset.
        
        Args:
            preprocessor: Instancia de AudioPreprocessor
            human_dir: Directorio con audios humanos
            spoof_dir: Directorio con audios spoof
            metadata_path: Ruta al CSV de metadatos (alternativa)
        """
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.human_dir = Path(human_dir) if human_dir else None
        self.spoof_dir = Path(spoof_dir) if spoof_dir else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
    
    def load_from_directories(
        self,
        max_samples_per_class: Optional[int] = None,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Carga dataset desde directorios human/ y spoof/.
        
        Args:
            max_samples_per_class: Límite de muestras por clase
            shuffle: Si mezclar los datos
            random_seed: Semilla para reproducibilidad
        
        Returns:
            X: Features (n_samples, fixed_length, n_mfcc)
            y: Labels (n_samples,) - 0=human, 1=spoof
            filenames: Lista de nombres de archivo
        """
        if not self.human_dir or not self.spoof_dir:
            raise ValueError("Se requieren human_dir y spoof_dir")
        
        np.random.seed(random_seed)
        
        features = []
        labels = []
        filenames = []
        
        # Extensiones de audio soportadas
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        # Cargar audios humanos (label = 0)
        human_files = [
            f for f in self.human_dir.iterdir()
            if f.suffix.lower() in audio_extensions
        ]
        
        if max_samples_per_class and len(human_files) > max_samples_per_class:
            human_files = list(np.random.choice(
                human_files, max_samples_per_class, replace=False
            ))
        
        logger.info(f"📂 Cargando {len(human_files)} audios humanos...")
        for file_path in tqdm(human_files, desc="Human"):
            try:
                feat = self.preprocessor.process_file(file_path)
                features.append(feat)
                labels.append(0)
                filenames.append(file_path.name)
            except Exception as e:
                logger.warning(f"Error procesando {file_path.name}: {e}")
        
        # Cargar audios spoof (label = 1)
        spoof_files = [
            f for f in self.spoof_dir.iterdir()
            if f.suffix.lower() in audio_extensions
        ]
        
        if max_samples_per_class and len(spoof_files) > max_samples_per_class:
            spoof_files = list(np.random.choice(
                spoof_files, max_samples_per_class, replace=False
            ))
        
        logger.info(f"📂 Cargando {len(spoof_files)} audios spoof...")
        for file_path in tqdm(spoof_files, desc="Spoof"):
            try:
                feat = self.preprocessor.process_file(file_path)
                features.append(feat)
                labels.append(1)
                filenames.append(file_path.name)
            except Exception as e:
                logger.warning(f"Error procesando {file_path.name}: {e}")
        
        # Convertir a arrays
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        # Shuffle
        if shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            filenames = [filenames[i] for i in indices]
        
        logger.info(f"✅ Dataset cargado: {len(X)} muestras")
        logger.info(f"   Human: {(y == 0).sum()}, Spoof: {(y == 1).sum()}")
        
        return X, y, filenames
    
    def load_from_metadata(
        self,
        audio_dir: Union[str, Path],
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Carga dataset desde archivo CSV de metadatos.
        
        El CSV debe tener columnas: filename, label
        donde label es 'bonafide' o 'spoof'
        
        Args:
            audio_dir: Directorio donde están los audios
            max_samples: Límite total de muestras
            shuffle: Si mezclar los datos
            random_seed: Semilla para reproducibilidad
        
        Returns:
            X, y, filenames
        """
        if not self.metadata_path:
            raise ValueError("Se requiere metadata_path")
        
        np.random.seed(random_seed)
        audio_dir = Path(audio_dir)
        
        # Leer metadata
        df = pd.read_csv(self.metadata_path)
        
        # Mapear labels
        label_map = {'bonafide': 0, 'human': 0, 'spoof': 1, 'fake': 1}
        df['label_int'] = df['label'].map(label_map)
        
        if shuffle:
            df = df.sample(frac=1, random_state=random_seed)
        
        if max_samples:
            df = df.head(max_samples)
        
        features = []
        labels = []
        filenames = []
        
        logger.info(f"📂 Cargando {len(df)} audios desde metadata...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
            file_path = audio_dir / row['filename']
            try:
                feat = self.preprocessor.process_file(file_path)
                features.append(feat)
                labels.append(row['label_int'])
                filenames.append(row['filename'])
            except Exception as e:
                logger.warning(f"Error procesando {row['filename']}: {e}")
        
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        logger.info(f"✅ Dataset cargado: {len(X)} muestras")
        logger.info(f"   Human: {(y == 0).sum()}, Spoof: {(y == 1).sum()}")
        
        return X, y, filenames


# ============================================
# Funciones de utilidad
# ============================================

def get_default_preprocessor() -> AudioPreprocessor:
    """Retorna un preprocesador con configuración por defecto"""
    from .config import AUDIO_CONFIG
    return AudioPreprocessor(
        sample_rate=AUDIO_CONFIG.sample_rate,
        n_mfcc=AUDIO_CONFIG.n_mfcc,
        n_fft=AUDIO_CONFIG.n_fft,
        hop_length=AUDIO_CONFIG.hop_length,
        n_mels=AUDIO_CONFIG.n_mels,
        fmin=AUDIO_CONFIG.fmin,
        fmax=AUDIO_CONFIG.fmax,
        fixed_length=AUDIO_CONFIG.fixed_length
    )


if __name__ == "__main__":
    # Test del preprocesador
    import sys
    
    print("="*60)
    print("🎵 TEST DE PREPROCESAMIENTO")
    print("="*60)
    
    preprocessor = AudioPreprocessor()
    
    print(f"\nConfiguración:")
    print(f"  Sample rate: {preprocessor.sample_rate} Hz")
    print(f"  N MFCC: {preprocessor.n_mfcc}")
    print(f"  N FFT: {preprocessor.n_fft}")
    print(f"  Hop length: {preprocessor.hop_length}")
    print(f"  Fixed length: {preprocessor.fixed_length} frames")
    print(f"  Output shape: ({preprocessor.fixed_length}, {preprocessor.n_mfcc})")
    
    # Test con audio sintético
    print("\n📊 Test con audio sintético:")
    test_audio = np.random.randn(preprocessor.sample_rate * 3).astype(np.float32)
    test_audio = test_audio / np.abs(test_audio).max()
    
    mfccs = preprocessor.extract_mfcc(test_audio)
    print(f"  Audio input: {test_audio.shape}")
    print(f"  MFCCs output: {mfccs.shape}")
    
    mfccs_fixed = preprocessor.pad_or_truncate(mfccs)
    print(f"  After pad/truncate: {mfccs_fixed.shape}")
    
    print("\n✅ Preprocesamiento funcionando correctamente")
