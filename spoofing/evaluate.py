"""
📊 Evaluación del Modelo de Detección de Spoofing
=================================================

Este módulo implementa evaluación exhaustiva del modelo incluyendo:
- Accuracy, Precision, Recall, F1
- AUC-ROC con curva ROC
- Matriz de confusión
- Análisis de errores

¿Por qué AUC es crítica en Spoofing?
====================================

1. ACCURACY PUEDE SER ENGAÑOSA
   - Con dataset balanceado (50% human, 50% spoof), un modelo
     que predice todo como "spoof" tiene 50% accuracy
   - Parece razonable pero es inútil

2. AUC MIDE CAPACIDAD DE RANKING
   - AUC = Probabilidad de que un spoof aleatorio tenga mayor
     score que un human aleatorio
   - AUC = 0.95 significa que el modelo discrimina correctamente
     el 95% de los pares (human, spoof)

3. INDEPENDIENTE DEL THRESHOLD
   - Accuracy depende del threshold de decisión
   - AUC evalúa el modelo a través de todos los thresholds
   - Permite elegir threshold óptimo en producción

4. ESTÁNDAR EN SEGURIDAD
   - ASVspoof challenge usa EER (Equal Error Rate)
   - EER se deriva de la curva ROC
   - La comunidad de spoofing usa AUC/EER como métrica principal

5. DIFERENCIA ENTRE FP Y FN
   - Falso Positivo: Audio humano clasificado como spoof
     → Usuario legítimo bloqueado (inconveniente)
   - Falso Negativo: Audio spoof clasificado como humano
     → Atacante pasa (breach de seguridad)
   - AUC permite analizar este trade-off
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score
)
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class SpoofingEvaluator:
    """
    Evaluador completo para modelos de detección de spoofing.
    
    Proporciona métricas detalladas y visualizaciones para
    analizar el rendimiento del modelo.
    """
    
    def __init__(
        self,
        model: Optional[keras.Model] = None,
        model_path: Optional[str] = None
    ):
        """
        Inicializa el evaluador.
        
        Args:
            model: Modelo Keras cargado
            model_path: Ruta al modelo guardado
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = keras.models.load_model(model_path)
        else:
            raise ValueError("Debe proporcionar model o model_path")
        
        self.y_true = None
        self.y_pred_proba = None
        self.y_pred = None
        self.threshold = 0.5
    
    def predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Genera predicciones para el conjunto de datos.
        
        Args:
            X: Features de entrada
            y: Labels verdaderas
            threshold: Umbral de decisión
        
        Returns:
            Diccionario con predicciones
        """
        self.y_true = y
        self.threshold = threshold
        
        # Predicciones de probabilidad
        self.y_pred_proba = self.model.predict(X, verbose=0).flatten()
        
        # Predicciones binarias
        self.y_pred = (self.y_pred_proba >= threshold).astype(int)
        
        return {
            'y_true': self.y_true,
            'y_pred_proba': self.y_pred_proba,
            'y_pred': self.y_pred
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas completas de evaluación.
        
        Returns:
            Diccionario con todas las métricas
        """
        if self.y_true is None:
            raise ValueError("Ejecute predict() primero")
        
        metrics = {
            # Métricas básicas
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'f1': f1_score(self.y_true, self.y_pred, zero_division=0),
            
            # AUC-ROC
            'auc_roc': roc_auc_score(self.y_true, self.y_pred_proba),
            
            # Average Precision (AUC-PR)
            'avg_precision': average_precision_score(self.y_true, self.y_pred_proba),
            
            # Threshold utilizado
            'threshold': self.threshold
        }
        
        # Equal Error Rate (EER)
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        metrics['eer'] = float(fpr[eer_idx])
        metrics['eer_threshold'] = float(thresholds[eer_idx])
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Calcula la matriz de confusión"""
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_classification_report(self) -> str:
        """Genera reporte de clasificación detallado"""
        return classification_report(
            self.y_true, self.y_pred,
            target_names=['Human', 'Spoof']
        )
    
    def find_optimal_threshold(
        self,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Encuentra el threshold óptimo para una métrica.
        
        Args:
            metric: 'f1', 'precision', 'recall', o 'eer'
        
        Returns:
            (threshold_óptimo, valor_métrica)
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_value = 0
        
        for thresh in thresholds:
            y_pred_temp = (self.y_pred_proba >= thresh).astype(int)
            
            if metric == 'f1':
                value = f1_score(self.y_true, y_pred_temp, zero_division=0)
            elif metric == 'precision':
                value = precision_score(self.y_true, y_pred_temp, zero_division=0)
            elif metric == 'recall':
                value = recall_score(self.y_true, y_pred_temp, zero_division=0)
            elif metric == 'eer':
                # Para EER, buscamos el threshold donde FPR ≈ FNR
                fp = ((y_pred_temp == 1) & (self.y_true == 0)).sum()
                fn = ((y_pred_temp == 0) & (self.y_true == 1)).sum()
                fpr = fp / (self.y_true == 0).sum()
                fnr = fn / (self.y_true == 1).sum()
                value = -abs(fpr - fnr)  # Negativo porque queremos minimizar
            else:
                raise ValueError(f"Métrica no soportada: {metric}")
            
            if value > best_value:
                best_value = value
                best_threshold = thresh
        
        return best_threshold, best_value
    
    def plot_roc_curve(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Genera la curva ROC.
        
        La curva ROC muestra el trade-off entre:
        - TPR (True Positive Rate): Spoofs correctamente detectados
        - FPR (False Positive Rate): Humans incorrectamente marcados
        
        Args:
            save_path: Ruta para guardar la figura
            figsize: Tamaño de la figura
        
        Returns:
            Figura de matplotlib
        """
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        auc = roc_auc_score(self.y_true, self.y_pred_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Curva ROC
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        
        # Línea diagonal (modelo aleatorio)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random (AUC = 0.5)')
        
        # Punto de EER
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        ax.scatter(fpr[eer_idx], tpr[eer_idx], c='green', s=100, zorder=5,
                   label=f'EER = {fpr[eer_idx]:.4f}')
        
        # Punto del threshold actual
        current_idx = np.argmin(np.abs(thresholds - self.threshold))
        ax.scatter(fpr[current_idx], tpr[current_idx], c='orange', s=100, zorder=5,
                   label=f'Threshold = {self.threshold:.2f}')
        
        ax.set_xlabel('False Positive Rate (Human → Spoof)', fontsize=12)
        ax.set_ylabel('True Positive Rate (Spoof detected)', fontsize=12)
        ax.set_title('ROC Curve - Spoofing Detection', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"📊 ROC curve guardada en: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Genera visualización de la matriz de confusión.
        
        Args:
            save_path: Ruta para guardar la figura
            figsize: Tamaño de la figura
        
        Returns:
            Figura de matplotlib
        """
        cm = self.get_confusion_matrix()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Matriz de confusión como heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        classes = ['Human', 'Spoof']
        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            ylabel='True label',
            xlabel='Predicted label',
            title=f'Confusion Matrix (threshold={self.threshold:.2f})'
        )
        
        # Rotar labels del eje X
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Añadir valores en las celdas
        thresh = cm.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"📊 Confusion matrix guardada en: {save_path}")
        
        return fig
    
    def plot_prediction_distribution(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Muestra la distribución de probabilidades por clase.
        
        Útil para entender cómo el modelo separa las clases.
        
        Args:
            save_path: Ruta para guardar
            figsize: Tamaño de figura
        
        Returns:
            Figura de matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Separar probabilidades por clase
        human_probs = self.y_pred_proba[self.y_true == 0]
        spoof_probs = self.y_pred_proba[self.y_true == 1]
        
        # Histogramas
        bins = np.linspace(0, 1, 50)
        ax.hist(human_probs, bins=bins, alpha=0.7, label='Human', color='green')
        ax.hist(spoof_probs, bins=bins, alpha=0.7, label='Spoof', color='red')
        
        # Línea de threshold
        ax.axvline(x=self.threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {self.threshold:.2f}')
        
        ax.set_xlabel('P(Spoof)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Prediction Distribution by Class', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def full_evaluation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: Optional[str] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Ejecuta evaluación completa y guarda resultados.
        
        Args:
            X: Features de entrada
            y: Labels verdaderas
            output_dir: Directorio para guardar resultados
            threshold: Umbral de decisión
        
        Returns:
            Diccionario con todos los resultados
        """
        # Generar predicciones
        self.predict(X, y, threshold)
        
        # Calcular métricas
        metrics = self.compute_metrics()
        
        # Report de clasificación
        report = self.get_classification_report()
        
        # Threshold óptimo
        opt_thresh, opt_f1 = self.find_optimal_threshold('f1')
        metrics['optimal_threshold_f1'] = opt_thresh
        metrics['optimal_f1'] = opt_f1
        
        # Guardar resultados
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar métricas como JSON
            with open(output_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Guardar reporte
            with open(output_dir / 'classification_report.txt', 'w') as f:
                f.write(report)
            
            # Guardar gráficas
            self.plot_roc_curve(save_path=str(output_dir / 'roc_curve.png'))
            self.plot_confusion_matrix(save_path=str(output_dir / 'confusion_matrix.png'))
            self.plot_prediction_distribution(save_path=str(output_dir / 'prediction_dist.png'))
            
            logger.info(f"📁 Resultados guardados en: {output_dir}")
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': self.get_confusion_matrix().tolist()
        }


def evaluate_model(
    model_path: str,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str = "outputs/evaluation",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Función de alto nivel para evaluar un modelo.
    
    Args:
        model_path: Ruta al modelo guardado
        X: Features de prueba
        y: Labels de prueba
        output_dir: Directorio de salida
        threshold: Umbral de decisión
    
    Returns:
        Resultados de la evaluación
    """
    evaluator = SpoofingEvaluator(model_path=model_path)
    return evaluator.full_evaluation(X, y, output_dir, threshold)


if __name__ == "__main__":
    print("="*70)
    print("📊 MÓDULO DE EVALUACIÓN DE SPOOFING")
    print("="*70)
    print("""
Este módulo proporciona evaluación exhaustiva para modelos de spoofing:

MÉTRICAS IMPLEMENTADAS:
- Accuracy: Proporción de predicciones correctas
- Precision: De los marcados como spoof, cuántos lo son realmente
- Recall: De los spoofs reales, cuántos detectamos
- F1-Score: Media armónica de precision y recall
- AUC-ROC: Área bajo la curva ROC (capacidad de ranking)
- EER: Equal Error Rate (FPR = FNR)
- Average Precision: Área bajo curva Precision-Recall

VISUALIZACIONES:
- Curva ROC con punto de EER
- Matriz de confusión
- Distribución de predicciones por clase

USO:
    from spoofing.evaluate import SpoofingEvaluator
    
    evaluator = SpoofingEvaluator(model_path='model.keras')
    results = evaluator.full_evaluation(X_test, y_test, 'outputs/eval')
""")
