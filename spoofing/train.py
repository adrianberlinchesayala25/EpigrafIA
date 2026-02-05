"""
🚀 Pipeline de Entrenamiento para Detección de Spoofing
========================================================

Este módulo implementa el pipeline completo de entrenamiento
para el detector de audio spoofing.

Características:
- División train/validation estratificada
- Callbacks para early stopping y LR scheduling
- Logging detallado y checkpoints
- Métricas de evaluación completas
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from .config import (
    AUDIO_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG
)
from .preprocessing import AudioPreprocessor, DatasetLoader
from .model import create_spoofing_detector, compile_model

logger = logging.getLogger(__name__)


class SpoofingTrainer:
    """
    Entrenador para el modelo de detección de spoofing.
    
    Gestiona el pipeline completo: carga de datos, entrenamiento,
    validación y guardado del modelo.
    """
    
    def __init__(
        self,
        human_dir: Optional[Path] = None,
        spoof_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa el entrenador.
        
        Args:
            human_dir: Directorio con audios humanos
            spoof_dir: Directorio con audios spoof
            output_dir: Directorio para guardar modelos
            config: Configuración personalizada
        """
        self.human_dir = human_dir or PATH_CONFIG.human_dir
        self.spoof_dir = spoof_dir or PATH_CONFIG.spoof_dir
        self.output_dir = output_dir or PATH_CONFIG.models_dir
        
        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración
        self.audio_config = AUDIO_CONFIG
        self.model_config = MODEL_CONFIG
        self.training_config = TRAINING_CONFIG
        
        # Sobrescribir con config personalizada
        if config:
            for key, value in config.items():
                if hasattr(self.training_config, key):
                    setattr(self.training_config, key, value)
        
        # Componentes
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.audio_config.sample_rate,
            n_mfcc=self.audio_config.n_mfcc,
            n_fft=self.audio_config.n_fft,
            hop_length=self.audio_config.hop_length,
            n_mels=self.audio_config.n_mels,
            fixed_length=self.audio_config.fixed_length
        )
        
        self.model = None
        self.history = None
        
        # Fijar semilla para reproducibilidad
        self._set_seeds(self.training_config.random_seed)
    
    def _set_seeds(self, seed: int):
        """Fija semillas para reproducibilidad"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def load_data(
        self,
        max_samples_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga y divide el dataset.
        
        Args:
            max_samples_per_class: Límite de muestras por clase
        
        Returns:
            X_train, X_val, y_train, y_val
        """
        logger.info("📂 Cargando dataset...")
        
        loader = DatasetLoader(
            preprocessor=self.preprocessor,
            human_dir=self.human_dir,
            spoof_dir=self.spoof_dir
        )
        
        X, y, filenames = loader.load_from_directories(
            max_samples_per_class=max_samples_per_class,
            shuffle=True,
            random_seed=self.training_config.random_seed
        )
        
        # División estratificada train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.training_config.validation_split,
            random_state=self.training_config.random_seed,
            stratify=y  # Mantener proporción de clases
        )
        
        logger.info(f"📊 División del dataset:")
        logger.info(f"   Train: {len(X_train)} ({(y_train==0).sum()} human, {(y_train==1).sum()} spoof)")
        logger.info(f"   Val: {len(X_val)} ({(y_val==0).sum()} human, {(y_val==1).sum()} spoof)")
        
        return X_train, X_val, y_train, y_val
    
    def create_model(self) -> keras.Model:
        """Crea y compila el modelo"""
        
        logger.info("🧠 Creando modelo...")
        
        input_shape = self.audio_config.input_shape
        
        self.model = create_spoofing_detector(
            input_shape=input_shape,
            conv_filters=self.model_config.conv_filters,
            conv_kernels=self.model_config.conv_kernels,
            dropout_rates=self.model_config.dropout_rates,
            dense_units=self.model_config.dense_units,
            dense_dropout=self.model_config.dense_dropout,
            leaky_alpha=self.model_config.leaky_alpha
        )
        
        self.model = compile_model(
            self.model,
            learning_rate=self.training_config.learning_rate
        )
        
        # Log summary
        total_params = self.model.count_params()
        logger.info(f"   Parámetros totales: {total_params:,}")
        
        return self.model
    
    def get_callbacks(self, run_name: str) -> list:
        """
        Crea callbacks para el entrenamiento.
        
        Callbacks incluidos:
        1. EarlyStopping: Detiene si val_loss no mejora
        2. ReduceLROnPlateau: Reduce LR si val_loss estanca
        3. ModelCheckpoint: Guarda mejor modelo
        4. TensorBoard: Logging para visualización
        
        Args:
            run_name: Nombre del experimento
        
        Returns:
            Lista de callbacks
        """
        callbacks = []
        
        # 1. Early Stopping
        # Detiene el entrenamiento si val_loss no mejora
        # patience=15 es suficiente para detectar convergencia
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.training_config.patience,
            min_delta=self.training_config.min_delta,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # 2. Reduce LR on Plateau
        # Reduce learning rate si val_loss no mejora por 5 epochs
        # Ayuda a escapar de mínimos locales
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.training_config.lr_factor,
            patience=self.training_config.lr_patience,
            min_lr=self.training_config.min_lr,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # 3. Model Checkpoint
        # Guarda el mejor modelo basado en val_auc
        checkpoint_path = self.output_dir / f"{run_name}_best.keras"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # 4. TensorBoard
        log_dir = PATH_CONFIG.logs_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        # 5. CSV Logger
        csv_path = self.output_dir / f"{run_name}_history.csv"
        csv_logger = keras.callbacks.CSVLogger(
            str(csv_path),
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        run_name: Optional[str] = None
    ) -> keras.callbacks.History:
        """
        Entrena el modelo.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            X_val: Features de validación
            y_val: Labels de validación
            run_name: Nombre del experimento
        
        Returns:
            Historia del entrenamiento
        """
        if self.model is None:
            self.create_model()
        
        # Nombre del experimento
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"spoofing_{timestamp}"
        
        logger.info(f"🚀 Iniciando entrenamiento: {run_name}")
        logger.info(f"   Epochs: {self.training_config.epochs}")
        logger.info(f"   Batch size: {self.training_config.batch_size}")
        logger.info(f"   Learning rate: {self.training_config.learning_rate}")
        
        # Callbacks
        callbacks = self.get_callbacks(run_name)
        
        # Calcular class weights si hay desbalance
        class_counts = np.bincount(y_train)
        total = len(y_train)
        class_weight = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        logger.info(f"   Class weights: {class_weight}")
        
        # Entrenamiento
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.training_config.epochs,
            batch_size=self.training_config.batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Guardar modelo final
        final_path = self.output_dir / f"{run_name}_final.keras"
        self.model.save(str(final_path))
        logger.info(f"💾 Modelo guardado en: {final_path}")
        
        return self.history
    
    def run_full_pipeline(
        self,
        max_samples_per_class: Optional[int] = None,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de entrenamiento.
        
        Args:
            max_samples_per_class: Límite de muestras por clase
            run_name: Nombre del experimento
        
        Returns:
            Diccionario con resultados
        """
        # 1. Cargar datos
        X_train, X_val, y_train, y_val = self.load_data(
            max_samples_per_class=max_samples_per_class
        )
        
        # 2. Crear modelo
        self.create_model()
        
        # 3. Entrenar
        history = self.train(X_train, y_train, X_val, y_val, run_name)
        
        # 4. Evaluar
        logger.info("\n📊 Evaluación final:")
        val_results = self.model.evaluate(X_val, y_val, verbose=0)
        
        metrics = {}
        for name, value in zip(self.model.metrics_names, val_results):
            metrics[f"val_{name}"] = float(value)
            logger.info(f"   {name}: {value:.4f}")
        
        return {
            'history': history.history,
            'metrics': metrics,
            'model_path': str(self.output_dir / f"{run_name}_best.keras")
        }


def train_spoofing_model(
    human_dir: str = "data/audios/human",
    spoof_dir: str = "data/audios/spoof",
    output_dir: str = "outputs/spoofing_models",
    max_samples: Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4
) -> Dict[str, Any]:
    """
    Función de alto nivel para entrenar el modelo de spoofing.
    
    Args:
        human_dir: Directorio con audios humanos
        spoof_dir: Directorio con audios spoof
        output_dir: Directorio para guardar modelos
        max_samples: Límite de muestras por clase
        epochs: Número de epochs
        batch_size: Tamaño de batch
        learning_rate: Tasa de aprendizaje
    
    Returns:
        Resultados del entrenamiento
    """
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Crear trainer
    trainer = SpoofingTrainer(
        human_dir=Path(human_dir),
        spoof_dir=Path(spoof_dir),
        output_dir=Path(output_dir),
        config={
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    )
    
    # Ejecutar pipeline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = trainer.run_full_pipeline(
        max_samples_per_class=max_samples,
        run_name=f"spoofing_{timestamp}"
    )
    
    return results


# ============================================
# Script principal
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entrenar modelo de detección de spoofing"
    )
    parser.add_argument(
        "--human-dir", type=str,
        default="data/audios/human",
        help="Directorio con audios humanos"
    )
    parser.add_argument(
        "--spoof-dir", type=str,
        default="data/audios/spoof",
        help="Directorio con audios spoof"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/spoofing_models",
        help="Directorio de salida"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Máximo de muestras por clase"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Número de epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Tamaño de batch"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🛡️ ENTRENAMIENTO DE DETECTOR DE SPOOFING")
    print("="*70)
    
    results = train_spoofing_model(
        human_dir=args.human_dir,
        spoof_dir=args.spoof_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Modelo guardado en: {results['model_path']}")
    print(f"Métricas finales: {results['metrics']}")
