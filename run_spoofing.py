#!/usr/bin/env python3
"""
🛡️ EpigrafIA - Detección de Audio Spoofing
==========================================

Script unificado para entrenar, evaluar e inferir con el modelo de spoofing.

Uso:
    # Entrenar modelo
    python run_spoofing.py train
    python run_spoofing.py train --epochs 50 --batch-size 64
    
    # Evaluar modelo
    python run_spoofing.py evaluate
    python run_spoofing.py evaluate --model outputs/spoofing_models/spoofing_best.keras
    
    # Inferencia en archivo
    python run_spoofing.py predict audio.wav
    python run_spoofing.py predict --dir audios/
    
    # Iniciar API
    python run_spoofing.py serve --port 8000
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
from pathlib import Path

# Agregar path del proyecto
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_train(args):
    """Entrenar modelo de spoofing"""
    from spoofing.train import train_spoofing_model
    
    print("\n" + "="*60)
    print("🛡️ ENTRENAMIENTO DE DETECTOR DE SPOOFING")
    print("="*60 + "\n")
    
    results = train_spoofing_model(
        human_dir=args.human_dir,
        spoof_dir=args.spoof_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"   Modelo: {results['model_path']}")
    for k, v in results['metrics'].items():
        print(f"   {k}: {v:.4f}")


def cmd_evaluate(args):
    """Evaluar modelo en test set"""
    from spoofing.evaluate import SpoofingEvaluator
    from spoofing.preprocessing import AudioPreprocessor, DatasetLoader
    from spoofing.config import AUDIO_CONFIG, PATH_CONFIG
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    print("\n" + "="*60)
    print("📊 EVALUACIÓN DEL MODELO")
    print("="*60 + "\n")
    
    # Cargar modelo
    model_path = args.model or (PATH_CONFIG.models_dir / "spoofing_best.keras")
    print(f"📂 Modelo: {model_path}")
    
    evaluator = SpoofingEvaluator(model_path=str(model_path))
    
    # Cargar datos
    preprocessor = AudioPreprocessor(
        sample_rate=AUDIO_CONFIG.sample_rate,
        n_mfcc=AUDIO_CONFIG.n_mfcc,
        n_fft=AUDIO_CONFIG.n_fft,
        hop_length=AUDIO_CONFIG.hop_length,
        fixed_length=AUDIO_CONFIG.fixed_length
    )
    
    loader = DatasetLoader(
        preprocessor=preprocessor,
        human_dir=PATH_CONFIG.human_dir,
        spoof_dir=PATH_CONFIG.spoof_dir
    )
    
    X, y, _ = loader.load_from_directories(shuffle=True, random_seed=42)
    
    # Usar 20% para test
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Evaluar
    metrics = evaluator.evaluate(X_test, y_test, threshold=args.threshold)
    
    print("\n" + "="*60)
    print("📈 RESULTADOS")
    print("="*60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
    
    # Generar gráficos si se pide
    if args.plots:
        output_dir = Path(args.output_dir or "outputs/spoofing_models")
        evaluator.plot_roc_curve(save_path=output_dir / "roc_curve.png")
        evaluator.plot_confusion_matrix(save_path=output_dir / "confusion_matrix.png")
        print(f"\n📊 Gráficos guardados en: {output_dir}")


def cmd_predict(args):
    """Inferencia en archivos de audio"""
    from spoofing.predict import SpoofingPredictor
    from spoofing.config import PATH_CONFIG
    import json
    
    print("\n" + "="*60)
    print("🔍 DETECCIÓN DE SPOOFING")
    print("="*60 + "\n")
    
    # Cargar modelo
    model_path = args.model or (PATH_CONFIG.models_dir / "spoofing_final.keras")
    if not Path(model_path).exists():
        model_path = PATH_CONFIG.models_dir / "spoofing_best.keras"
    
    predictor = SpoofingPredictor(
        model_path=str(model_path),
        threshold=args.threshold
    )
    
    # Procesar archivos
    if args.dir:
        # Directorio
        audio_dir = Path(args.dir)
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.flac"))
        print(f"📂 Procesando {len(audio_files)} archivos de {audio_dir}\n")
    elif args.audio:
        audio_files = [Path(args.audio)]
    else:
        print("❌ Error: Especifica un archivo o directorio")
        return
    
    results = []
    for audio_path in audio_files:
        result = predictor.predict_file(str(audio_path))
        results.append(result)
        
        # Mostrar resultado
        icon = "🔴" if result.get('is_spoof') else "🟢"
        label = "SPOOF" if result.get('is_spoof') else "HUMAN"
        conf = result.get('confidence', 0) * 100
        print(f"   {icon} {audio_path.name}: {label} ({conf:.1f}%)")
    
    # Resumen
    spoof_count = sum(1 for r in results if r.get('is_spoof'))
    human_count = len(results) - spoof_count
    
    print(f"\n📈 Resumen: {human_count} humanos, {spoof_count} spoof")
    
    # JSON si se pide
    if args.json:
        print("\n" + json.dumps(results, indent=2, ensure_ascii=False))


def cmd_serve(args):
    """Iniciar API FastAPI"""
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 INICIANDO API DE SPOOFING")
    print("="*60 + "\n")
    
    uvicorn.run(
        "spoofing.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def main():
    parser = argparse.ArgumentParser(
        description="🛡️ EpigrafIA - Detección de Audio Spoofing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # ========== TRAIN ==========
    train_parser = subparsers.add_parser("train", help="Entrenar modelo")
    train_parser.add_argument("--human-dir", default="data/spoofing/humano", help="Dir con audios humanos")
    train_parser.add_argument("--spoof-dir", default="data/spoofing/spoof", help="Dir con audios spoof")
    train_parser.add_argument("--output-dir", default="outputs/spoofing_models", help="Dir de salida")
    train_parser.add_argument("--max-samples", type=int, help="Máx muestras por clase")
    train_parser.add_argument("--epochs", type=int, default=100, help="Número de epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # ========== EVALUATE ==========
    eval_parser = subparsers.add_parser("evaluate", help="Evaluar modelo")
    eval_parser.add_argument("--model", help="Ruta al modelo .keras")
    eval_parser.add_argument("--threshold", type=float, default=0.5, help="Threshold de decisión")
    eval_parser.add_argument("--plots", action="store_true", help="Generar gráficos")
    eval_parser.add_argument("--output-dir", help="Dir para gráficos")
    
    # ========== PREDICT ==========
    predict_parser = subparsers.add_parser("predict", help="Inferencia en archivos")
    predict_parser.add_argument("audio", nargs="?", help="Archivo de audio")
    predict_parser.add_argument("--dir", "-d", help="Directorio con audios")
    predict_parser.add_argument("--model", help="Ruta al modelo .keras")
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    predict_parser.add_argument("--json", action="store_true", help="Output JSON")
    
    # ========== SERVE ==========
    serve_parser = subparsers.add_parser("serve", help="Iniciar API")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Puerto")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload")
    
    args = parser.parse_args()
    
    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
