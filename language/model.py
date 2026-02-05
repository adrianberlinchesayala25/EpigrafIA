"""
🧠 Advanced Language Detection Models
======================================
State-of-the-art CNN architectures for language classification.

Features:
- Squeeze-and-Excitation (SE) attention blocks
- Residual connections
- Multi-scale feature extraction
- EfficientNet-inspired design
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from typing import Tuple, Optional

from .config import LanguageConfig, DEFAULT_CONFIG, LANGUAGE_LABELS


# ============================================
# Building Blocks
# ============================================

def squeeze_excitation_block(x: tf.Tensor, ratio: int = 16, 
                              name: str = "se") -> tf.Tensor:
    """
    Squeeze-and-Excitation attention block.
    
    Channel attention mechanism that adaptively recalibrates
    channel-wise feature responses.
    
    Args:
        x: Input tensor of shape (batch, time, channels)
        ratio: Reduction ratio for bottleneck
        name: Name prefix for layers
    
    Returns:
        Attention-weighted tensor
    """
    channels = x.shape[-1]
    
    # Squeeze: Global average pooling
    se = layers.GlobalAveragePooling1D(name=f"{name}_squeeze")(x)
    
    # Excitation: Two FC layers with bottleneck
    se = layers.Dense(channels // ratio, activation='relu', 
                      name=f"{name}_fc1")(se)
    se = layers.Dense(channels, activation='sigmoid', 
                      name=f"{name}_fc2")(se)
    
    # Reshape for multiplication
    se = layers.Reshape((1, channels), name=f"{name}_reshape")(se)
    
    # Scale: multiply input by attention weights
    return layers.Multiply(name=f"{name}_scale")([x, se])


def conv_block(x: tf.Tensor, filters: int, kernel_size: int = 3,
               strides: int = 1, use_bn: bool = True,
               activation: str = 'leaky_relu',
               dropout: float = 0.0, name: str = "conv") -> tf.Tensor:
    """
    Convolutional block with BatchNorm, activation, and dropout.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Size of convolutional kernel
        strides: Stride of convolution
        use_bn: Whether to use BatchNormalization
        activation: Activation function ('relu', 'leaky_relu', 'swish')
        dropout: Dropout rate
        name: Name prefix
    
    Returns:
        Processed tensor
    """
    x = layers.Conv1D(filters, kernel_size, strides=strides, 
                      padding='same', use_bias=not use_bn,
                      name=f"{name}_conv")(x)
    
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    
    if activation == 'leaky_relu':
        x = layers.LeakyReLU(negative_slope=0.1, name=f"{name}_act")(x)
    elif activation == 'swish':
        x = layers.Activation('swish', name=f"{name}_act")(x)
    elif activation == 'relu':
        x = layers.Activation('relu', name=f"{name}_act")(x)
    
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    
    return x


def residual_block(x: tf.Tensor, filters: int, kernel_size: int = 3,
                   use_se: bool = True, se_ratio: int = 16,
                   dropout: float = 0.1, name: str = "res") -> tf.Tensor:
    """
    Residual block with optional SE attention.
    
    Structure:
    input -> Conv -> BN -> Act -> Conv -> BN -> SE -> Add -> Act
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size for convolutions
        use_se: Whether to use SE attention
        se_ratio: SE reduction ratio
        dropout: Dropout rate
        name: Name prefix
    
    Returns:
        Residual output tensor
    """
    shortcut = x
    
    # First conv
    out = conv_block(x, filters, kernel_size, 
                     dropout=dropout, name=f"{name}_conv1")
    
    # Second conv (no activation yet)
    out = layers.Conv1D(filters, kernel_size, padding='same',
                        name=f"{name}_conv2")(out)
    out = layers.BatchNormalization(name=f"{name}_bn2")(out)
    
    # SE attention
    if use_se:
        out = squeeze_excitation_block(out, ratio=se_ratio, name=f"{name}_se")
    
    # Match dimensions for residual connection
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same',
                                 name=f"{name}_shortcut")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)
    
    # Add residual and apply activation
    out = layers.Add(name=f"{name}_add")([out, shortcut])
    out = layers.LeakyReLU(negative_slope=0.1, name=f"{name}_out_act")(out)
    
    if dropout > 0:
        out = layers.Dropout(dropout, name=f"{name}_out_drop")(out)
    
    return out


# ============================================
# Basic CNN Model (Original)
# ============================================

def create_language_model(
    input_shape: Tuple[int, int] = (94, 120),
    num_classes: int = 4,
    dropout_rate: float = 0.4
) -> keras.Model:
    """
    Create basic CNN model for language detection (original architecture).
    
    Args:
        input_shape: Shape of input features (time_steps, features)
        num_classes: Number of languages to classify
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='audio_input')
    
    # Conv Block 1
    x = layers.Conv1D(64, 5, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # Conv Block 2
    x = layers.Conv1D(128, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate * 0.75)(x)
    
    # Conv Block 3
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Conv Block 4
    x = layers.Conv1D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', 
                           name='language_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LanguageDetectionCNN')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model


# ============================================
# Advanced CNN Model with SE Attention
# ============================================

def create_advanced_language_model(
    config: LanguageConfig = None
) -> keras.Model:
    """
    Create advanced CNN model with SE attention and residual connections.
    
    Architecture:
    - Multi-scale initial convolutions
    - 4 residual blocks with SE attention
    - Multi-head global pooling
    - Dense classifier with label smoothing ready
    
    Args:
        config: Language detection configuration
    
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    input_shape = config.input_shape
    num_classes = config.num_classes
    dropout = config.dropout_rate
    base_filters = config.base_filters
    
    inputs = layers.Input(shape=input_shape, name='audio_input')
    
    # ========== Initial Multi-Scale Feature Extraction ==========
    # Branch 1: Fine features (kernel 3)
    branch1 = layers.Conv1D(base_filters // 2, 3, padding='same',
                            name='init_conv_k3')(inputs)
    
    # Branch 2: Medium features (kernel 5)
    branch2 = layers.Conv1D(base_filters // 2, 5, padding='same',
                            name='init_conv_k5')(inputs)
    
    # Branch 3: Coarse features (kernel 7)
    branch3 = layers.Conv1D(base_filters // 2, 7, padding='same',
                            name='init_conv_k7')(inputs)
    
    # Concatenate multi-scale features
    x = layers.Concatenate(name='init_concat')([branch1, branch2, branch3])
    x = layers.BatchNormalization(name='init_bn')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='init_act')(x)
    
    # Spatial dropout for 1D sequences
    x = layers.SpatialDropout1D(config.spatial_dropout, name='init_spatial_drop')(x)
    
    # ========== Residual Blocks with SE Attention ==========
    filter_sizes = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]
    # [64, 128, 256, 512]
    
    for i, filters in enumerate(filter_sizes):
        # Residual block with SE
        x = residual_block(
            x, filters,
            kernel_size=3 if i >= 2 else 5,
            use_se=config.use_se_attention,
            se_ratio=config.se_ratio,
            dropout=dropout * (0.5 + 0.25 * i),  # Increasing dropout
            name=f'res_block_{i+1}'
        )
        
        # Downsample (except last block)
        if i < len(filter_sizes) - 1:
            x = layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}')(x)
    
    # ========== Multi-Head Global Pooling ==========
    # Average pooling
    avg_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Max pooling (captures prominent features)
    max_pool = layers.GlobalMaxPooling1D(name='global_max_pool')(x)
    
    # Combine pooling features
    x = layers.Concatenate(name='pool_concat')([avg_pool, max_pool])
    x = layers.BatchNormalization(name='pool_bn')(x)
    
    # ========== Dense Classifier ==========
    x = layers.Dense(512, name='fc1')(x)
    x = layers.BatchNormalization(name='fc1_bn')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='fc1_act')(x)
    x = layers.Dropout(dropout, name='fc1_drop')(x)
    
    x = layers.Dense(256, name='fc2')(x)
    x = layers.BatchNormalization(name='fc2_bn')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='fc2_act')(x)
    x = layers.Dropout(dropout * 0.75, name='fc2_drop')(x)
    
    x = layers.Dense(128, name='fc3')(x)
    x = layers.BatchNormalization(name='fc3_bn')(x)
    x = layers.LeakyReLU(negative_slope=0.1, name='fc3_act')(x)
    x = layers.Dropout(dropout * 0.5, name='fc3_drop')(x)
    
    # ========== Output ==========
    outputs = layers.Dense(num_classes, activation='softmax', 
                           name='language_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='AdvancedLanguageCNN')
    
    return model


# ============================================
# Loss Functions
# ============================================

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for class imbalance.
    
    Focuses training on hard examples by down-weighting easy ones.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 label_smoothing: float = 0.0, 
                 from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        # Cast to same dtype
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Get number of classes
        num_classes = tf.shape(y_pred)[-1]
        
        # Check if y_true is already one-hot (from mixup) or sparse
        # If y_true has shape (batch,) or (batch, 1), convert to one-hot
        # If y_true has shape (batch, num_classes), use as-is
        y_true_shape = tf.shape(y_true)
        y_true_rank = len(y_true.shape)
        
        if y_true_rank == 1 or (y_true_rank == 2 and y_true.shape[-1] == 1):
            # Sparse labels - convert to one-hot
            y_true = tf.squeeze(y_true, axis=-1) if y_true_rank == 2 else y_true
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
            y_true = tf.cast(y_true, tf.float32)
        
        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            num_classes_f = tf.cast(num_classes, tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes_f
        
        # Compute focal loss
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        
        focal_loss = self.alpha * focal_weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing,
            'from_logits': self.from_logits,
        })
        return config


def get_loss_function(config: LanguageConfig = None):
    """Get appropriate loss function based on config"""
    if config is None:
        config = DEFAULT_CONFIG
    
    if config.use_focal_loss:
        return FocalLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing
        )
    else:
        return keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )


# ============================================
# Learning Rate Schedules
# ============================================

def get_cosine_schedule(total_steps: int, warmup_steps: int = 0,
                        base_lr: float = 1e-3, min_lr: float = 1e-6):
    """
    Cosine annealing learning rate schedule with optional warmup.
    
    Args:
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
    
    Returns:
        Learning rate schedule function
    """
    def schedule(step):
        if step < warmup_steps:
            # Linear warmup
            return base_lr * step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + tf.cos(progress * 3.14159))
    
    return schedule


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + Cosine decay learning rate schedule"""
    
    def __init__(self, base_lr: float, total_steps: int, 
                 warmup_steps: int = 0, min_lr: float = 1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr = float(min_lr)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.constant(self.warmup_steps, dtype=tf.float32)
        total_steps = tf.constant(self.total_steps, dtype=tf.float32)
        
        # Warmup phase
        warmup_lr = self.base_lr * (step / tf.maximum(warmup_steps, 1.0))
        
        # Cosine decay phase
        decay_steps = tf.maximum(total_steps - warmup_steps, 1.0)
        decay_progress = (step - warmup_steps) / decay_steps
        decay_progress = tf.maximum(0.0, tf.minimum(1.0, decay_progress))
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                    (1.0 + tf.cos(decay_progress * 3.14159265359))
        
        # Select phase
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)
    
    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'min_lr': self.min_lr,
        }


# ============================================
# Model Factory
# ============================================

def build_model(config: LanguageConfig = None, 
                compile_model: bool = True) -> keras.Model:
    """
    Build model based on configuration.
    
    Args:
        config: Language detection configuration
        compile_model: Whether to compile the model
    
    Returns:
        Keras model (compiled or not)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Create model
    if config.model_type == "basic":
        model = create_language_model(
            input_shape=config.input_shape,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate
        )
    else:
        model = create_advanced_language_model(config)
    
    if compile_model:
        # Get loss
        loss = get_loss_function(config)
        
        # Get optimizer with weight decay
        optimizer = keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            clipnorm=1.0  # Gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
    
    return model


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    import numpy as np
    
    print("\n" + "="*60)
    print("🧠 LANGUAGE DETECTION MODEL TEST")
    print("="*60)
    
    config = LanguageConfig()
    
    # Test basic model
    print("\n📊 Basic Model:")
    basic_model = create_language_model()
    basic_model.summary()
    
    # Test advanced model
    print("\n📊 Advanced Model:")
    advanced_model = create_advanced_language_model(config)
    advanced_model.summary()
    
    # Test inference
    print("\n🧪 Testing inference...")
    test_input = np.random.randn(2, *config.input_shape).astype(np.float32)
    
    output = advanced_model.predict(test_input, verbose=0)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Predictions: {output}")
    print(f"   Predicted classes: {output.argmax(axis=1)}")
    
    # Count parameters
    total_params = advanced_model.count_params()
    print(f"\n📈 Total parameters: {total_params:,}")
    
    print("\n✅ All model tests passed!")
