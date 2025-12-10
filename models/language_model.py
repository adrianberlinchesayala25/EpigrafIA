"""
🗣️ Language Detection Model
============================
CNN model architecture for language classification

Supports 4 languages:
- Español (Spanish)
- Inglés (English)
- Francés (French)
- Alemán (German)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from keras import layers, Model
from keras.models import Sequential
import keras
from typing import Tuple, Optional


def create_language_model(
    input_shape: Tuple[int, int] = (94, 120),
    num_classes: int = 4,
    dropout_rate: float = 0.4
) -> keras.Model:
    """
    Create CNN model for language detection
    
    Architecture:
    - 4 Conv1D blocks with BatchNorm and Dropout
    - Global Average Pooling
    - Dense layers with softmax output
    
    Args:
        input_shape: Shape of input features (time_steps, features)
                     Default: (94 time frames, 120 MFCC features)
        num_classes: Number of languages to classify
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')
    
    # ========== CONV BLOCK 1 ==========
    x = layers.Conv1D(64, 5, padding='same', activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)  # LeakyReLU to prevent dying neurons
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # ========== CONV BLOCK 2 ==========
    x = layers.Conv1D(128, 5, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate * 0.75)(x)
    
    # ========== CONV BLOCK 3 ==========
    x = layers.Conv1D(256, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # ========== CONV BLOCK 4 ==========
    x = layers.Conv1D(512, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # ========== POOLING ==========
    x = layers.GlobalAveragePooling1D()(x)
    
    # ========== DENSE LAYERS ==========
    x = layers.Dense(256, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)  # LeakyReLU
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)  # LeakyReLU
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # ========== OUTPUT ==========
    outputs = layers.Dense(num_classes, activation='softmax', name='language_output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='LanguageDetectionCNN')
    
    # Compile with label smoothing to prevent overconfidence
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model


def create_language_model_lstm(
    input_shape: Tuple[int, int] = (94, 120),
    num_classes: int = 4,
    dropout_rate: float = 0.4
) -> keras.Model:
    """
    Alternative LSTM-based model for language detection
    
    Args:
        input_shape: Shape of input features (time_steps, features)
        num_classes: Number of languages to classify
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='language_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LanguageDetectionLSTM')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """Get model summary as string"""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return '\n'.join(lines)


# ============================================
# Labels and Metadata
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


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    # Create model
    model = create_language_model()
    
    print("\n" + "="*60)
    print("📊 LANGUAGE DETECTION MODEL")
    print("="*60)
    model.summary()
    
    # Test with random input
    import numpy as np
    test_input = np.random.randn(1, 94, 120).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    
    print(f"\n✅ Test prediction shape: {output.shape}")
    print(f"   Probabilities: {output[0]}")
    print(f"   Predicted class: {output.argmax()} ({LANGUAGE_LABELS[output.argmax()]})")
