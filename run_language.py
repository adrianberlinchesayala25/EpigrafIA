#!/usr/bin/env python3
"""
🗣️ EpigrafIA - Detector de Idioma SIMPLE
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import argparse
import numpy as np
from pathlib import Path

# ============================================
# CONFIG
# ============================================
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 40
LANGUAGES = ["es", "en", "fr", "de"]
LANG_NAMES = {"es": "Español", "en": "English", "fr": "Français", "de": "Deutsch"}

DATA_DIRS = {
    "es": Path("data/Common Voice/Audios Español/clips"),
    "en": Path("data/Common Voice/Audios Ingles/clips"),
    "fr": Path("data/Common Voice/Audios Frances/clips"),
    "de": Path("data/Common Voice/Audios Aleman/clips"),
}

OUTPUT_DIR = Path("outputs/language_models")


def extract_mfcc(path: str, augment: bool = False) -> np.ndarray:
    """Extrae MFCCs + deltas (más features = mejor accuracy)"""
    import librosa
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
        target = int(SAMPLE_RATE * DURATION)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        
        # Augmentation simple (solo en training)
        if augment:
            # Ruido aleatorio
            if np.random.random() < 0.3:
                noise = np.random.randn(len(y)) * 0.005
                y = y + noise
            # Pitch shift leve
            if np.random.random() < 0.2:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-1, 1))
        
        # MFCC + deltas (3x más features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenar: (n_mfcc*3, time)
        features = np.concatenate([mfcc, delta, delta2], axis=0)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        return features.T  # (time, 120)
    except:
        return None


def load_data(max_samples=300):
    """Carga dataset"""
    X, y = [], []
    
    for idx, lang in enumerate(LANGUAGES):
        lang_dir = DATA_DIRS[lang]
        if not lang_dir.exists():
            print(f"⚠️  {lang_dir} no existe")
            continue
        
        files = list(lang_dir.glob("*.mp3"))[:max_samples]
        print(f"📂 {LANG_NAMES[lang]}: {len(files)} archivos")
        
        for f in files:
            mfcc = extract_mfcc(str(f))
            if mfcc is not None:
                X.append(mfcc)
                y.append(idx)
    
    return np.array(X), np.array(y)


def create_model(input_shape):
    """Modelo CNN mejorado pero rápido"""
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Bloque 1
        layers.Conv1D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        # Bloque 2
        layers.Conv1D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        # Bloque 3
        layers.Conv1D(256, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling1D(),
        
        # Clasificador
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train(args):
    """Entrenar"""
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    
    print("\n🗣️ ENTRENAMIENTO DETECTOR DE IDIOMA (v2)\n")
    
    # Cargar
    X, y = load_data(args.samples)
    print(f"\n✅ Total: {len(X)} muestras, shape: {X.shape}\n")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y)
    
    # Modelo
    model = create_model(X_train.shape[1:])
    model.summary()
    
    # Entrenar
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(OUTPUT_DIR / "language_best.keras"),
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
    )
    
    # Guardar
    model.save(str(OUTPUT_DIR / "language_final.keras"))
    
    # Copiar a models_trained
    import shutil
    dest = Path("outputs/models_trained")
    dest.mkdir(exist_ok=True)
    shutil.copy(OUTPUT_DIR / "language_best.keras", dest / "language_model_best.keras")
    
    val_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Completado! Accuracy: {val_acc*100:.1f}%")
    print(f"📁 Modelo: {dest / 'language_model_best.keras'}")


def predict(args):
    """Predecir"""
    import tensorflow as tf
    
    model_path = Path("outputs/models_trained/language_model_best.keras")
    if not model_path.exists():
        model_path = OUTPUT_DIR / "language_best.keras"
    
    if not model_path.exists():
        print("❌ No hay modelo. Ejecuta: python run_language.py train")
        return
    
    model = tf.keras.models.load_model(str(model_path))
    
    mfcc = extract_mfcc(args.audio)
    if mfcc is None:
        print("❌ Error procesando audio")
        return
    
    probs = model.predict(np.expand_dims(mfcc, 0), verbose=0)[0]
    idx = np.argmax(probs)
    
    flags = {"es": "🇪🇸", "en": "🇬🇧", "fr": "🇫🇷", "de": "🇩🇪"}
    lang = LANGUAGES[idx]
    
    print(f"\n{flags[lang]} {LANG_NAMES[lang]} ({probs[idx]*100:.1f}%)\n")


def main():
    parser = argparse.ArgumentParser(description="🗣️ Detector de Idioma")
    sub = parser.add_subparsers(dest="cmd")
    
    # Train
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--samples", type=int, default=300)
    
    # Predict
    p = sub.add_parser("predict")
    p.add_argument("audio")
    
    args = parser.parse_args()
    
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
