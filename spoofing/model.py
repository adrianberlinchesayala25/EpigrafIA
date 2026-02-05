"""
🧠 Arquitectura CNN 1D para Detección de Audio Spoofing
========================================================

Este módulo implementa una CNN 1D optimizada para distinguir entre
audio humano real (bonafide) y audio generado por IA (spoof).

FUNDAMENTOS DE LA ARQUITECTURA
==============================

¿Por qué CNN 1D para Spoofing?
------------------------------
1. Los MFCCs tienen correlación temporal que las CNNs capturan eficientemente
2. Las CNNs detectan patrones locales (artefactos de síntesis)
3. Menor tendencia al overfitting que Transformers con datasets pequeños
4. Invariancia a la posición temporal de los artefactos

Diseño de la Arquitectura:
--------------------------

1. BLOQUES CONVOLUCIONALES (64 → 128 → 256 → 512)
   - Progresión de filtros: Captura características cada vez más abstractas
   - Primeros bloques: Patrones acústicos básicos (formantes, pitch)
   - Últimos bloques: Patrones de alto nivel (naturalidad, artefactos)

2. KERNELS (5, 5, 3, 3)
   - Kernels más grandes al inicio: Capturan contexto temporal amplio
   - Kernels más pequeños después: Refinan detalles finos
   - Esto es crucial para detectar microartefactos de síntesis

3. BATCH NORMALIZATION
   - Estabiliza el entrenamiento
   - Actúa como regularización
   - Permite learning rates más altos

4. LEAKY RELU (α=0.1)
   - Evita "dying neurons" comunes con ReLU estándar
   - Mantiene gradientes pequeños para valores negativos
   - Mejor para detección de patrones sutiles

5. DROPOUT PROGRESIVO (0.2 → 0.3 → 0.4 → 0.4)
   - Menos dropout inicial: Preserva información básica
   - Más dropout en capas profundas: Previene overfitting
   - Crítico para generalización a nuevos generadores de IA

6. GLOBAL AVERAGE POOLING (vs Flatten)
   - Reduce parámetros dramáticamente
   - Proporciona invariancia a la longitud
   - Mejor generalización

7. CAPAS DENSAS (256 → 128 → 1)
   - Reducción gradual hacia la decisión binaria
   - Suficiente capacidad sin overfitting
   - Dropout 0.5 entre densas

8. SALIDA SIGMOIDE
   - Clasificación binaria: P(spoof | audio)
   - Interpretable como probabilidad

¿Por qué NO usar Transformers aquí?
-----------------------------------
- Dataset de 20k es pequeño para Transformers
- Mayor riesgo de overfitting
- CNNs son más eficientes para patrones locales
- Los artefactos de spoofing son locales, no requieren atención global
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, regularizers
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_spoofing_detector(
    input_shape: Tuple[int, int] = (400, 40),
    conv_filters: Tuple[int, ...] = (64, 128, 256, 512),
    conv_kernels: Tuple[int, ...] = (5, 5, 3, 3),
    dropout_rates: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.4),
    dense_units: Tuple[int, ...] = (256, 128),
    dense_dropout: float = 0.5,
    leaky_alpha: float = 0.1,
    l2_reg: float = 1e-4
) -> keras.Model:
    """
    Crea el modelo CNN 1D para detección de spoofing.
    
    Arquitectura optimizada para detectar artefactos sutiles en audio
    generado por IA vs voz humana natural.
    
    Args:
        input_shape: Shape de entrada (time_steps, n_mfcc)
        conv_filters: Filtros por bloque convolucional
        conv_kernels: Tamaño de kernel por bloque
        dropout_rates: Dropout por bloque convolucional
        dense_units: Unidades en capas densas
        dense_dropout: Dropout entre capas densas
        leaky_alpha: Alpha para LeakyReLU
        l2_reg: Regularización L2
    
    Returns:
        Modelo Keras compilado
    """
    
    # ============================================
    # INPUT LAYER
    # ============================================
    inputs = layers.Input(shape=input_shape, name='mfcc_input')
    x = inputs
    
    # ============================================
    # BLOQUES CONVOLUCIONALES
    # ============================================
    # Cada bloque: Conv1D → BatchNorm → LeakyReLU → MaxPool → Dropout
    
    for i, (filters, kernel, dropout) in enumerate(zip(
        conv_filters, conv_kernels, dropout_rates
    )):
        block_name = f"conv_block_{i+1}"
        
        # Convolución 1D
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel,
            padding='same',
            use_bias=False,  # BatchNorm incluye bias
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{block_name}_conv"
        )(x)
        
        # Batch Normalization
        x = layers.BatchNormalization(name=f"{block_name}_bn")(x)
        
        # Activación LeakyReLU
        x = layers.LeakyReLU(negative_slope=leaky_alpha, name=f"{block_name}_lrelu")(x)
        
        # MaxPooling (excepto en el último bloque)
        if i < len(conv_filters) - 1:
            x = layers.MaxPooling1D(pool_size=2, name=f"{block_name}_pool")(x)
        
        # Dropout
        x = layers.Dropout(dropout, name=f"{block_name}_dropout")(x)
    
    # ============================================
    # GLOBAL AVERAGE POOLING
    # ============================================
    # Reduce (batch, time, filters) → (batch, filters)
    # Ventajas:
    # - Invariante a la longitud temporal
    # - Reduce parámetros significativamente
    # - Mejor generalización que Flatten
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # ============================================
    # CAPAS DENSAS
    # ============================================
    for i, units in enumerate(dense_units):
        dense_name = f"dense_{i+1}"
        
        x = layers.Dense(
            units,
            use_bias=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{dense_name}_linear"
        )(x)
        
        x = layers.BatchNormalization(name=f"{dense_name}_bn")(x)
        x = layers.LeakyReLU(negative_slope=leaky_alpha, name=f"{dense_name}_lrelu")(x)
        x = layers.Dropout(dense_dropout, name=f"{dense_name}_dropout")(x)
    
    # ============================================
    # CAPA DE SALIDA
    # ============================================
    # Sigmoide para clasificación binaria
    # Output: P(spoof | audio) ∈ [0, 1]
    outputs = layers.Dense(
        1,
        activation='sigmoid',
        name='spoof_output'
    )(x)
    
    # ============================================
    # CREAR MODELO
    # ============================================
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name='SpoofingDetectorCNN'
    )
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-4
) -> keras.Model:
    """
    Compila el modelo con configuración optimizada para spoofing.
    
    Configuración:
    - Optimizer: Adam con lr=1e-4 (conservador para convergencia estable)
    - Loss: Binary Crossentropy (clasificación binaria)
    - Métricas: Accuracy + AUC
    
    ¿Por qué AUC es crítica para Spoofing?
    --------------------------------------
    1. Accuracy puede ser engañosa con clases balanceadas pero
       diferentes costos de error (falso positivo vs falso negativo)
    
    2. AUC mide la capacidad de ranking del modelo:
       - Un modelo con AUC=0.95 significa que hay 95% de probabilidad
         de que un audio spoof aleatorio tenga score mayor que uno humano
    
    3. AUC es invariante al threshold de decisión:
       - Permite elegir el threshold óptimo post-entrenamiento
       - Ej: En producción, podemos preferir alta precision (menos falsos positivos)
    
    4. En aplicaciones de seguridad, AUC es el estándar de la industria
       (ASVspoof challenge usa EER derivado de curva ROC)
    
    Args:
        model: Modelo Keras a compilar
        learning_rate: Tasa de aprendizaje
    
    Returns:
        Modelo compilado
    """
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc', curve='ROC'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """Obtiene el summary del modelo como string"""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return '\n'.join(lines)


def count_parameters(model: keras.Model) -> dict:
    """Cuenta parámetros del modelo"""
    trainable = sum(
        tf.reduce_prod(w.shape).numpy() 
        for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.reduce_prod(w.shape).numpy() 
        for w in model.non_trainable_weights
    )
    return {
        'trainable': int(trainable),
        'non_trainable': int(non_trainable),
        'total': int(trainable + non_trainable)
    }


# ============================================
# Modelo alternativo con Residual Connections
# ============================================

def create_spoofing_detector_residual(
    input_shape: Tuple[int, int] = (400, 40)
) -> keras.Model:
    """
    Versión con conexiones residuales para mejor flujo de gradientes.
    
    Útil si el modelo base tiene problemas de convergencia.
    """
    
    inputs = layers.Input(shape=input_shape, name='mfcc_input')
    
    # Proyección inicial
    x = layers.Conv1D(64, 1, padding='same')(inputs)
    
    # Bloque residual 1
    shortcut = x
    x = layers.Conv1D(64, 5, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv1D(64, 5, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Bloque residual 2
    shortcut = layers.Conv1D(128, 1, padding='same')(x)
    x = layers.Conv1D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv1D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Bloque residual 3
    shortcut = layers.Conv1D(256, 1, padding='same')(x)
    x = layers.Conv1D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv1D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.4)(x)
    
    # Global pooling y clasificador
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='spoof_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='SpoofingDetectorResidual')
    
    return model


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    import numpy as np
    
    print("="*70)
    print("🧠 SPOOFING DETECTOR - ARQUITECTURA CNN 1D")
    print("="*70)
    
    # Crear modelo
    model = create_spoofing_detector()
    model = compile_model(model)
    
    print("\n📊 RESUMEN DEL MODELO:")
    print("-"*70)
    model.summary()
    
    # Contar parámetros
    params = count_parameters(model)
    print(f"\n📈 PARÁMETROS:")
    print(f"   Entrenables: {params['trainable']:,}")
    print(f"   No entrenables: {params['non_trainable']:,}")
    print(f"   Total: {params['total']:,}")
    
    # Test con entrada aleatoria
    print("\n🧪 TEST DE INFERENCIA:")
    test_input = np.random.randn(4, 400, 40).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Predictions: {output.flatten()}")
    
    print("\n✅ Modelo creado correctamente")
    
    # Explicación de la arquitectura
    print("\n" + "="*70)
    print("📖 JUSTIFICACIÓN DE LA ARQUITECTURA")
    print("="*70)
    print("""
Esta arquitectura CNN 1D está diseñada específicamente para detección de
audio spoofing (deepfakes de voz). Los principios clave son:

1. JERARQUÍA DE CARACTERÍSTICAS
   - Bloques iniciales (64, 128 filtros): Detectan patrones acústicos
     básicos como formantes, transiciones y micromodulaciones.
   - Bloques finales (256, 512 filtros): Capturan patrones de alto nivel
     que distinguen voz natural de sintética.

2. KERNELS DECRECIENTES (5→5→3→3)
   - Kernels grandes al inicio capturan contexto temporal amplio
   - Kernels pequeños al final refinan detalles finos
   - Los artefactos de TTS suelen ser locales (clicks, discontinuidades)

3. DROPOUT PROGRESIVO (0.2→0.3→0.4→0.4)
   - Menos regularización en capas iniciales preserva información
   - Mayor regularización en capas profundas previene overfitting
   - Crucial para generalizar a nuevos sistemas TTS/VC

4. GLOBAL AVERAGE POOLING
   - Reduce dimensionalidad sin perder información
   - Proporciona invariancia a posición temporal
   - Mejora generalización vs Flatten

5. AUC COMO MÉTRICA PRINCIPAL
   - Accuracy sola puede ser engañosa
   - AUC mide capacidad de ranking independiente del threshold
   - Estándar en competiciones de spoofing (ASVspoof)
""")
