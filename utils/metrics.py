"""Metrics for model evaluation"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from typing import Dict, Tuple
import torch


class ModelMetrics:
    """Calculate various evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_probs: np.ndarray = None,
                         average: str = 'weighted') -> Dict:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities (for ROC-AUC)
            average: Average method for precision/recall/f1
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        
        # Add ROC-AUC if probabilities provided
        if y_probs is not None:
            try:
                num_classes = len(np.unique(y_true))
                if num_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr',
                                                      average='weighted')
            except Exception as e:
                print(f"Could not calculate ROC-AUC: {e}")
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray,
                                 y_pred: np.ndarray) -> str:
        """
        Get detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred)
    
    @staticmethod
    def get_per_class_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray) -> Dict:
        """
        Get per-class metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of per-class metrics
        """
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        return {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
        }


class AverageMeter:
    """Average meter for tracking metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update metric
        
        Args:
            val: Current value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.avg:.4f}"


def get_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate accuracy from model output
    
    Args:
        output: Model output logits
        target: Target labels
        
    Returns:
        Accuracy value
    """
    _, pred = torch.max(output, 1)
    correct = (pred == target).sum().item()
    return correct / target.size(0)
