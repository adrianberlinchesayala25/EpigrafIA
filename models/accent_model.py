"""
🎭 Accent Detection Model
=========================
CNN model architecture for accent/dialect classification

Supports 8 accents:
- España (Spain Spanish)
- México (Mexican Spanish)
- UK (British English)
- USA (American English)
- Francia (France French)
- Quebec (Canadian French)
- Alemania (German German)
- Austria (Austrian German)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from keras import layers, Model
import keras
from typing import Tuple, Optional


def create_accent_model(
    input_shape: Tuple[int, int] = (94, 120),
    num_classes: int = 8,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Create CNN model for accent detection
    
    This model is deeper than the language model because accent
    detection requires more subtle feature recognition.
    
    Architecture:
    - 5 Conv1D blocks with BatchNorm and Dropout
    - Residual connections in later blocks
    - Global Average Pooling
    - Dense layers with softmax output
    
    Args:
        input_shape: Shape of input features (time_steps, features)
                     Default: (94 time frames, 120 MFCC features)
        num_classes: Number of accents to classify
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')
    
    # ========== CONV BLOCK 1 ==========
    x = layers.Conv1D(64, 7, padding='same', activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate * 0.3)(x)
    
    # ========== CONV BLOCK 2 ==========
    x = layers.Conv1D(128, 5, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # ========== CONV BLOCK 3 ==========
    x = layers.Conv1D(256, 5, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    
    # ========== CONV BLOCK 4 with Residual ==========
    conv4 = layers.Conv1D(512, 3, padding='same', activation=None)(x)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.ReLU()(conv4)
    
    conv4b = layers.Conv1D(512, 3, padding='same', activation=None)(conv4)
    conv4b = layers.BatchNormalization()(conv4b)
    
    # Residual connection (project x to match dimensions)
    x_proj = layers.Conv1D(512, 1, padding='same')(x)
    x = layers.Add()([conv4b, x_proj])
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # ========== CONV BLOCK 5 ==========
    x = layers.Conv1D(512, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # ========== POOLING ==========
    # Use both global average and max pooling
    gap = layers.GlobalAveragePooling1D()(x)
    gmp = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    # ========== DENSE LAYERS ==========
    x = layers.Dense(512, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)
    
    x = layers.Dense(128, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # ========== OUTPUT ==========
    outputs = layers.Dense(num_classes, activation='softmax', name='accent_output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='AccentDetectionCNN')
    
    # Compile with label smoothing for better generalization
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model


# ============================================
# Labels and Metadata
# ============================================

ACCENT_LABELS = {
    0: 'España',
    1: 'México',
    2: 'UK',
    3: 'USA',
    4: 'Francia',
    5: 'Quebec',
    6: 'Alemania',
    7: 'Austria'
}

ACCENT_BY_LANGUAGE = {
    'es': ['España', 'México'],
    'en': ['UK', 'USA'],
    'fr': ['Francia', 'Quebec'],
    'de': ['Alemania', 'Austria']
}


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    # Create model
    model = create_accent_model()
    
    print("\n" + "="*60)
    print("📊 ACCENT DETECTION MODEL")
    print("="*60)
    model.summary()
    
    # Test with random input
    import numpy as np
    test_input = np.random.randn(1, 94, 120).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    
    print(f"\n✅ Test prediction shape: {output.shape}")
    print(f"   Probabilities: {output[0]}")
    print(f"   Predicted class: {output.argmax()} ({ACCENT_LABELS[output.argmax()]})")