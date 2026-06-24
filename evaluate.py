"""Evaluation script for TENG CNN model"""

import torch
import numpy as np
from pathlib import Path

from config import DEVICE, MODEL_DIR, SIGNAL_LENGTH, NUM_CLASSES
from models.cnn_model import TENGCNN1D
from data.dataset import TENGDataset
from utils.metrics import ModelMetrics
from utils.visualization import MetricsVisualizer
from train import Trainer, set_seed
from data.dataset import get_data_loaders, create_dummy_dataset


def evaluate_model(model_path: str,
                  test_signals: np.ndarray,
                  test_labels: np.ndarray,
                  device: str = 'cpu'):
    """
    Evaluate trained model
    
    Args:
        model_path: Path to model checkpoint
        test_signals: Test signals
        test_labels: Test labels
        device: Device to evaluate on
    """
    # Load model
    model = TENGCNN1D(
        input_channels=1,
        num_classes=NUM_CLASSES
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create test dataset and loader
    test_dataset = TENGDataset(test_signals, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    # Predict
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = ModelMetrics.calculate_metrics(
        all_labels, all_preds, all_probs
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Per-class metrics
    per_class_metrics = ModelMetrics.get_per_class_metrics(all_labels, all_preds)
    print("\n" + "="*50)
    print("PER-CLASS METRICS")
    print("="*50)
    for class_idx in range(NUM_CLASSES):
        print(f"Class {class_idx}:")
        print(f"  Precision: {per_class_metrics['precision'][class_idx]:.4f}")
        print(f"  Recall: {per_class_metrics['recall'][class_idx]:.4f}")
        print(f"  F1: {per_class_metrics['f1'][class_idx]:.4f}")
    
    # Confusion matrix
    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    cm = ModelMetrics.get_confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(ModelMetrics.get_classification_report(all_labels, all_preds))
    
    # Visualization
    print("\nGenerating visualizations...")
    MetricsVisualizer.plot_confusion_matrix(all_labels, all_preds)
    MetricsVisualizer.plot_metrics_bar(metrics)
    
    return metrics, all_preds, all_probs, all_labels


def main():
    """Main evaluation function"""
    set_seed(42)
    
    # Create dummy test data
    print("Creating test data...")
    _, test_signals = create_dummy_dataset(
        n_samples=1000,
        signal_length=SIGNAL_LENGTH,
        num_classes=NUM_CLASSES
    )
    signals, labels = create_dummy_dataset(
        n_samples=200,
        signal_length=SIGNAL_LENGTH,
        num_classes=NUM_CLASSES
    )
    
    # Evaluate
    device = DEVICE if torch.cuda.is_available() else 'cpu'
    model_path = MODEL_DIR / "best_model.pt"
    
    if model_path.exists():
        print(f"Loading model from {model_path}")
        metrics, preds, probs, labels_true = evaluate_model(
            str(model_path),
            signals,
            labels,
            device=device
        )
    else:
        print(f"Model not found at {model_path}")
        print("Please train a model first using train.py")


if __name__ == "__main__":
    main()
