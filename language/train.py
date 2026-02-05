"""
🏋️ Language Detection Training Pipeline
=========================================
Advanced training with all state-of-the-art techniques.

Features:
- Data loading with stratified sampling
- Mixup and CutMix augmentation
- Focal loss with label smoothing
- Cosine annealing learning rate
- Early stopping with best model checkpointing
- K-Fold cross validation option
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm
import random
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .config import LanguageConfig, LANGUAGE_FOLDERS, LANGUAGE_LABELS
from .preprocessing import LanguagePreprocessor, AudioAugmenter
from .model import build_model, WarmupCosineDecay, FocalLoss


class MixupGenerator(keras.utils.Sequence):
    """
    Data generator with Mixup and CutMix augmentation.
    
    Implements on-the-fly augmentation for training.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 config: LanguageConfig,
                 shuffle: bool = True,
                 augment: bool = True):
        self.X = X
        self.y = y
        self.config = config
        self.shuffle = shuffle
        self.augment = augment
        self.batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.augmenter = AudioAugmenter(config)
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    
    def __len__(self) -> int:
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices].copy()
        
        if self.augment:
            X_batch, y_batch = self._apply_augmentation(X_batch, y_batch)
        else:
            # Convert to one-hot for consistency
            y_batch = keras.utils.to_categorical(y_batch, self.num_classes)
        
        return X_batch, y_batch
    
    def _apply_augmentation(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Mixup or CutMix augmentation"""
        batch_size = len(X)
        
        # Convert labels to one-hot
        y_onehot = keras.utils.to_categorical(y, self.num_classes)
        
        # Apply SpecAugment to all samples
        for i in range(batch_size):
            X[i] = self.augmenter.spec_augment(X[i])
        
        # Randomly choose between Mixup and CutMix
        if self.config.use_mixup or self.config.use_cutmix:
            use_cutmix = self.config.use_cutmix and random.random() < self.config.cutmix_prob
            
            # Shuffle indices for mixing
            indices = np.random.permutation(batch_size)
            
            X_mixed = np.zeros_like(X)
            y_mixed = np.zeros_like(y_onehot)
            
            for i in range(batch_size):
                j = indices[i]
                
                if use_cutmix:
                    X_mixed[i], y_mixed[i] = self.augmenter.cutmix(
                        X[i], y[i], X[j], y[j],
                        alpha=self.config.cutmix_alpha,
                        num_classes=self.num_classes
                    )
                elif self.config.use_mixup and random.random() < 0.5:
                    X_mixed[i], y_mixed[i] = self.augmenter.mixup(
                        X[i], y[i], X[j], y[j],
                        alpha=self.config.mixup_alpha,
                        num_classes=self.num_classes
                    )
                else:
                    X_mixed[i] = X[i]
                    y_mixed[i] = y_onehot[i]
            
            return X_mixed, y_mixed
        
        return X, y_onehot
    
    def on_epoch_end(self):
        """Shuffle indices at end of epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


class LanguageTrainer:
    """
    Advanced trainer for language detection model.
    
    Implements:
    - Stratified data splitting
    - Class weighting for imbalanced data
    - Mixup/CutMix augmentation
    - Cosine annealing with warmup
    - K-Fold cross validation
    """
    
    def __init__(self, config: LanguageConfig = None):
        self.config = config or LanguageConfig()
        self.preprocessor = LanguagePreprocessor(self.config)
        self.model = None
        self.history = None
        
        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mixed precision if enabled
        if self.config.use_mixed_precision:
            try:
                policy = keras.mixed_precision.Policy('mixed_float16')
                keras.mixed_precision.set_global_policy(policy)
                print("🚀 Mixed precision (float16) enabled")
            except Exception as e:
                print(f"⚠️ Mixed precision not available: {e}")
    
    def load_dataset(self, num_augmentations: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from Common Voice folders.
        
        Returns:
            X: Features array of shape (samples, time_frames, features)
            y: Labels array of shape (samples,)
        """
        print("\n" + "="*60)
        print("📂 LOADING DATASET")
        print("="*60)
        
        X = []
        y = []
        
        for lang_folder, label in LANGUAGE_FOLDERS.items():
            lang_path = self.config.data_dir / lang_folder / "clips"
            
            if not lang_path.exists():
                print(f"⚠️ Folder not found: {lang_path}")
                continue
            
            # Get audio files
            audio_files = list(lang_path.glob("*.mp3"))
            
            # Shuffle and take subset
            random.shuffle(audio_files)
            audio_files = audio_files[:self.config.samples_per_language]
            
            lang_name = LANGUAGE_LABELS[label]
            print(f"\n📁 {lang_name}: {len(audio_files)} files")
            
            count = 0
            for audio_file in tqdm(audio_files, desc=f"   Processing"):
                # Extract multiple augmented versions
                features_list = self.preprocessor.extract_multiple_augmented(
                    audio_file, 
                    num_augmentations=num_augmentations
                )
                
                for features in features_list:
                    X.append(features)
                    y.append(label)
                    count += 1
            
            print(f"   📊 Total samples: {count}")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        print(f"\n✅ Dataset loaded:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights with optional boosting"""
        # Base weights from sklearn
        weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        
        # Apply manual boost factors
        if self.config.use_class_weights:
            for cls, boost in self.config.class_weight_boost.items():
                if cls < len(weights):
                    weights[cls] *= boost
        
        class_weights = {i: w for i, w in enumerate(weights)}
        
        return class_weights
    
    def get_callbacks(self, fold: int = 0) -> List[keras.callbacks.Callback]:
        """Get training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                str(self.config.output_dir / f"language_best_fold{fold}_{timestamp}.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                str(self.config.log_dir / f"training_log_fold{fold}_{timestamp}.csv")
            ),
        ]
        
        # Learning rate schedule
        if self.config.lr_schedule == "reduce_on_plateau":
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience,
                    min_lr=self.config.min_learning_rate,
                    verbose=1
                )
            )
        
        return callbacks
    
    def train(self, X: np.ndarray = None, y: np.ndarray = None) -> keras.Model:
        """
        Train language detection model.
        
        Args:
            X: Optional preloaded features
            y: Optional preloaded labels
        
        Returns:
            Trained Keras model
        """
        print("\n" + "="*60)
        print("🚀 LANGUAGE DETECTION TRAINING")
        print("="*60)
        
        # Load data if not provided
        if X is None or y is None:
            X, y = self.load_dataset()
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(self.config.validation_split + self.config.test_split),
            stratify=y,
            random_state=self.config.random_seed
        )
        
        val_ratio = self.config.validation_split / (self.config.validation_split + self.config.test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            stratify=y_temp,
            random_state=self.config.random_seed
        )
        
        print(f"\n📊 Data split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        print(f"\n⚖️ Class weights: {class_weights}")
        
        # Calculate training steps for LR schedule
        steps_per_epoch = len(X_train) // self.config.batch_size
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        
        # Build model with cosine schedule
        print("\n🧠 Building model...")
        self.model = build_model(self.config, compile_model=False)
        
        # Create optimizer with LR schedule
        if self.config.lr_schedule == "cosine" or self.config.lr_schedule == "warmup_cosine":
            lr_schedule = WarmupCosineDecay(
                base_lr=self.config.learning_rate,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr=self.config.min_learning_rate
            )
        else:
            lr_schedule = self.config.learning_rate
        
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.config.weight_decay,
            clipnorm=1.0
        )
        
        # Get loss function
        if self.config.use_focal_loss:
            loss = FocalLoss(
                gamma=self.config.focal_gamma,
                label_smoothing=self.config.label_smoothing
            )
        else:
            loss = keras.losses.CategoricalCrossentropy(
                label_smoothing=self.config.label_smoothing
            )
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        self.model.summary()
        
        # Create data generators
        train_gen = MixupGenerator(
            X_train, y_train,
            config=self.config,
            shuffle=True,
            augment=True
        )
        
        # Validation data (no augmentation)
        y_val_onehot = keras.utils.to_categorical(y_val, self.config.num_classes)
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train!
        print("\n" + "="*60)
        print("🏋️ TRAINING")
        print("="*60)
        
        self.history = self.model.fit(
            train_gen,
            validation_data=(X_val, y_val_onehot),
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_path = self.config.output_dir / "language_model_final.keras"
        self.model.save(str(final_path))
        print(f"\n✅ Final model saved to {final_path}")
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        return self.model
    
    def train_kfold(self, X: np.ndarray = None, y: np.ndarray = None) -> List[keras.Model]:
        """
        Train with K-Fold cross validation.
        
        Returns:
            List of trained models (one per fold)
        """
        print("\n" + "="*60)
        print(f"🔄 K-FOLD CROSS VALIDATION ({self.config.n_folds} folds)")
        print("="*60)
        
        # Load data if not provided
        if X is None or y is None:
            X, y = self.load_dataset()
        
        # Initialize K-Fold
        kfold = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_seed
        )
        
        models = []
        histories = []
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\n{'='*60}")
            print(f"📂 FOLD {fold + 1}/{self.config.n_folds}")
            print(f"{'='*60}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"   Train: {len(X_train)} | Val: {len(X_val)}")
            
            # Build fresh model
            model = build_model(self.config, compile_model=False)
            
            # Compile
            steps_per_epoch = len(X_train) // self.config.batch_size
            total_steps = steps_per_epoch * self.config.epochs
            warmup_steps = steps_per_epoch * self.config.warmup_epochs
            
            lr_schedule = WarmupCosineDecay(
                base_lr=self.config.learning_rate,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr=self.config.min_learning_rate
            )
            
            optimizer = keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=self.config.weight_decay,
                clipnorm=1.0
            )
            
            loss = FocalLoss(
                gamma=self.config.focal_gamma,
                label_smoothing=self.config.label_smoothing
            ) if self.config.use_focal_loss else keras.losses.CategoricalCrossentropy(
                label_smoothing=self.config.label_smoothing
            )
            
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            
            # Create generator
            train_gen = MixupGenerator(
                X_train, y_train,
                config=self.config,
                shuffle=True,
                augment=True
            )
            
            y_val_onehot = keras.utils.to_categorical(y_val, self.config.num_classes)
            
            # Train
            callbacks = self.get_callbacks(fold=fold)
            
            history = model.fit(
                train_gen,
                validation_data=(X_val, y_val_onehot),
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            val_loss, val_acc = model.evaluate(X_val, y_val_onehot, verbose=0)
            print(f"\n📊 Fold {fold+1} - Val Accuracy: {val_acc*100:.2f}%")
            
            models.append(model)
            histories.append(history)
            scores.append(val_acc)
        
        # Print summary
        print("\n" + "="*60)
        print("📈 K-FOLD SUMMARY")
        print("="*60)
        for i, score in enumerate(scores):
            print(f"   Fold {i+1}: {score*100:.2f}%")
        print(f"\n   Mean: {np.mean(scores)*100:.2f}% ± {np.std(scores)*100:.2f}%")
        
        return models


# ============================================
# Main
# ============================================

def train_language_model(config: LanguageConfig = None, use_kfold: bool = False):
    """Main training function"""
    if config is None:
        config = LanguageConfig()
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 GPU detected: {gpus}")
        # Memory growth
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("💻 Running on CPU (training will be slower)")
    
    trainer = LanguageTrainer(config)
    
    if use_kfold or config.use_kfold:
        models = trainer.train_kfold()
        return models[0]  # Return best model
    else:
        model = trainer.train()
        return model


if __name__ == "__main__":
    config = LanguageConfig()
    model = train_language_model(config)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
