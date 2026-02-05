"""
📊 Language Detection Evaluation
=================================
Comprehensive evaluation with metrics, plots, and analysis.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

from .config import LanguageConfig, LANGUAGE_LABELS
from .preprocessing import LanguagePreprocessor, AudioAugmenter


class LanguageEvaluator:
    """
    Evaluator for language detection model.
    
    Features:
    - Standard metrics (accuracy, F1, precision, recall)
    - Confusion matrix visualization
    - Per-class analysis
    - Test-Time Augmentation (TTA)
    - Confidence analysis
    """
    
    def __init__(self, model: keras.Model, config: LanguageConfig = None):
        self.model = model
        self.config = config or LanguageConfig()
        self.preprocessor = LanguagePreprocessor(self.config)
        self.augmenter = AudioAugmenter(self.config)
    
    def predict_with_tta(self, X: np.ndarray, 
                         num_augmentations: int = None) -> np.ndarray:
        """
        Predict with Test-Time Augmentation.
        
        Applies multiple augmentations and averages predictions.
        
        Args:
            X: Input features of shape (samples, time, features)
            num_augmentations: Number of augmented versions
            
        Returns:
            Averaged predictions of shape (samples, num_classes)
        """
        if num_augmentations is None:
            num_augmentations = self.config.tta_augmentations
        
        all_predictions = []
        
        # Original prediction
        pred = self.model.predict(X, verbose=0)
        all_predictions.append(pred)
        
        # Augmented predictions
        for _ in range(num_augmentations - 1):
            X_aug = np.array([
                self.augmenter.spec_augment(x.copy()) for x in X
            ])
            pred = self.model.predict(X_aug, verbose=0)
            all_predictions.append(pred)
        
        # Average predictions
        avg_predictions = np.mean(all_predictions, axis=0)
        
        return avg_predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 use_tta: bool = None, verbose: bool = True) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            use_tta: Whether to use TTA (default from config)
            verbose: Print results
            
        Returns:
            Dictionary with all metrics
        """
        if use_tta is None:
            use_tta = self.config.use_tta
        
        if verbose:
            print("\n" + "="*60)
            print("📊 MODEL EVALUATION")
            print("="*60)
        
        # Get predictions
        if use_tta:
            if verbose:
                print(f"\n🔄 Using Test-Time Augmentation ({self.config.tta_augmentations} augmentations)")
            y_proba = self.predict_with_tta(X_test)
        else:
            y_proba = self.model.predict(X_test, verbose=0)
        
        y_pred = np.argmax(y_proba, axis=1)
        y_conf = np.max(y_proba, axis=1)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        # Per-class metrics
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        target_names = [LANGUAGE_LABELS[i] for i in range(self.config.num_classes)]
        report = classification_report(y_test, y_pred, target_names=target_names)
        
        # Confidence analysis
        correct_mask = y_pred == y_test
        avg_conf_correct = y_conf[correct_mask].mean() if correct_mask.sum() > 0 else 0
        avg_conf_incorrect = y_conf[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'f1_per_class': {LANGUAGE_LABELS[i]: f1_per_class[i] for i in range(len(f1_per_class))},
            'confusion_matrix': cm,
            'classification_report': report,
            'avg_confidence_correct': avg_conf_correct,
            'avg_confidence_incorrect': avg_conf_incorrect,
            'predictions': y_pred,
            'probabilities': y_proba,
            'confidences': y_conf,
        }
        
        if verbose:
            print(f"\n📈 Overall Metrics:")
            print(f"   Accuracy: {accuracy*100:.2f}%")
            print(f"   F1 Macro: {f1_macro*100:.2f}%")
            print(f"   F1 Weighted: {f1_weighted*100:.2f}%")
            print(f"   Precision: {precision*100:.2f}%")
            print(f"   Recall: {recall*100:.2f}%")
            
            print(f"\n📊 Per-Class F1 Scores:")
            for lang, f1 in results['f1_per_class'].items():
                print(f"   {lang}: {f1*100:.2f}%")
            
            print(f"\n🎯 Confidence Analysis:")
            print(f"   Avg confidence (correct): {avg_conf_correct*100:.2f}%")
            print(f"   Avg confidence (incorrect): {avg_conf_incorrect*100:.2f}%")
            
            print(f"\n📋 Classification Report:")
            print(report)
            
            print(f"\n🔢 Confusion Matrix:")
            self._print_confusion_matrix(cm, target_names)
        
        return results
    
    def _print_confusion_matrix(self, cm: np.ndarray, labels: List[str]):
        """Print confusion matrix in readable format"""
        # Header
        header = "Pred→  " + "  ".join([f"{l[:4]:>6}" for l in labels])
        print(header)
        print("-" * len(header))
        
        for i, row in enumerate(cm):
            row_str = f"{labels[i][:4]:>6} |" + "  ".join([f"{v:>6}" for v in row])
            print(row_str)
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                              save_path: str = None,
                              show: bool = True):
        """Plot confusion matrix heatmap"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            
            labels = [LANGUAGE_LABELS[i] for i in range(self.config.num_classes)]
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(
                cm_normalized, 
                annot=True, 
                fmt='.2%',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels
            )
            
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix - Language Detection')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   Saved to {save_path}")
            
            if show:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("⚠️ matplotlib/seaborn not available for plotting")
    
    def plot_training_history(self, history, save_path: str = None,
                              show: bool = True):
        """Plot training history (loss and accuracy)"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Loss
            axes[0].plot(history.history['loss'], label='Train Loss')
            axes[0].plot(history.history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy
            axes[1].plot(history.history['accuracy'], label='Train Acc')
            axes[1].plot(history.history['val_accuracy'], label='Val Acc')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   Saved to {save_path}")
            
            if show:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("⚠️ matplotlib not available for plotting")
    
    def analyze_errors(self, X_test: np.ndarray, y_test: np.ndarray,
                       y_pred: np.ndarray, y_conf: np.ndarray,
                       top_n: int = 10) -> Dict:
        """Analyze prediction errors"""
        errors_mask = y_pred != y_test
        error_indices = np.where(errors_mask)[0]
        
        if len(error_indices) == 0:
            return {'num_errors': 0, 'error_rate': 0.0}
        
        # Get error details
        errors = []
        for idx in error_indices:
            errors.append({
                'index': int(idx),
                'true_label': LANGUAGE_LABELS[y_test[idx]],
                'pred_label': LANGUAGE_LABELS[y_pred[idx]],
                'confidence': float(y_conf[idx])
            })
        
        # Sort by confidence (most confident errors are most concerning)
        errors = sorted(errors, key=lambda x: x['confidence'], reverse=True)
        
        # Error distribution by true class
        error_by_true = {}
        for i in range(self.config.num_classes):
            mask = y_test[error_indices] == i
            error_by_true[LANGUAGE_LABELS[i]] = int(mask.sum())
        
        # Error distribution by predicted class
        error_by_pred = {}
        for i in range(self.config.num_classes):
            mask = y_pred[error_indices] == i
            error_by_pred[LANGUAGE_LABELS[i]] = int(mask.sum())
        
        # Most common confusions
        confusion_pairs = {}
        for e in errors:
            pair = f"{e['true_label']} → {e['pred_label']}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        confusion_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'num_errors': len(error_indices),
            'error_rate': len(error_indices) / len(y_test),
            'top_errors': errors[:top_n],
            'errors_by_true_class': error_by_true,
            'errors_by_pred_class': error_by_pred,
            'most_common_confusions': confusion_pairs[:5],
            'avg_error_confidence': np.mean([e['confidence'] for e in errors])
        }
    
    def save_results(self, results: Dict, filepath: str = None):
        """Save evaluation results to JSON"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.config.log_dir / f"evaluation_{timestamp}.json"
        
        # Convert numpy arrays to lists
        results_serializable = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_serializable[k] = v.tolist()
            elif isinstance(v, dict):
                results_serializable[k] = {
                    str(kk): vv.tolist() if isinstance(vv, np.ndarray) else vv
                    for kk, vv in v.items()
                }
            else:
                results_serializable[k] = v
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n💾 Results saved to {filepath}")


def evaluate_model(model_path: str = None, 
                   test_data: Tuple[np.ndarray, np.ndarray] = None,
                   config: LanguageConfig = None):
    """
    Standalone evaluation function.
    
    Args:
        model_path: Path to saved model
        test_data: Tuple of (X_test, y_test)
        config: Configuration
    """
    if config is None:
        config = LanguageConfig()
    
    if model_path is None:
        model_path = config.output_dir / "language_model_final.keras"
    
    print(f"\n📂 Loading model from {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    
    evaluator = LanguageEvaluator(model, config)
    
    if test_data is None:
        print("⚠️ No test data provided. Please provide (X_test, y_test)")
        return None
    
    X_test, y_test = test_data
    
    results = evaluator.evaluate(X_test, y_test, use_tta=config.use_tta)
    
    # Plot confusion matrix
    try:
        evaluator.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=str(config.log_dir / "confusion_matrix.png"),
            show=False
        )
    except Exception as e:
        print(f"⚠️ Could not plot confusion matrix: {e}")
    
    # Save results
    evaluator.save_results(results)
    
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 LANGUAGE DETECTION EVALUATION")
    print("="*60)
    print("\nUsage: evaluate_model(model_path, (X_test, y_test))")
