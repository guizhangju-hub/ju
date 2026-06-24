"""Visualization utilities"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Optional, Tuple


class SignalVisualizer:
    """Visualize sensor signals and analysis results"""
    
    @staticmethod
    def plot_signal(signal: np.ndarray,
                   sampling_rate: int = 1000,
                   title: str = "Signal",
                   figsize: Tuple = (12, 4)):
        """
        Plot 1D signal
        
        Args:
            signal: Input signal
            sampling_rate: Sampling rate
            title: Plot title
            figsize: Figure size
        """
        time = np.arange(len(signal)) / sampling_rate
        
        plt.figure(figsize=figsize)
        plt.plot(time, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_multiple_signals(signals: list,
                            labels: list = None,
                            sampling_rate: int = 1000,
                            figsize: Tuple = (12, 8)):
        """
        Plot multiple signals
        
        Args:
            signals: List of signals
            labels: List of signal labels
            sampling_rate: Sampling rate
            figsize: Figure size
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=figsize)
        
        if n_signals == 1:
            axes = [axes]
        
        for i, signal in enumerate(signals):
            time = np.arange(len(signal)) / sampling_rate
            axes[i].plot(time, signal)
            axes[i].set_ylabel('Amplitude')
            if labels:
                axes[i].set_title(labels[i])
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_fft(frequency: np.ndarray,
                magnitude: np.ndarray,
                title: str = "FFT Spectrum",
                figsize: Tuple = (12, 4)):
        """
        Plot FFT spectrum
        
        Args:
            frequency: Frequency array
            magnitude: Magnitude array
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(frequency, magnitude)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_signal_before_after(original: np.ndarray,
                                processed: np.ndarray,
                                sampling_rate: int = 1000,
                                figsize: Tuple = (12, 6)):
        """
        Compare original and processed signals
        
        Args:
            original: Original signal
            processed: Processed signal
            sampling_rate: Sampling rate
            figsize: Figure size
        """
        time = np.arange(len(original)) / sampling_rate
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        axes[0].plot(time, original)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Original Signal')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time, processed)
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Processed Signal')
        axes[1].set_xlabel('Time (s)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class MetricsVisualizer:
    """Visualize training and evaluation metrics"""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: list = None,
                             figsize: Tuple = (8, 6)):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names for labels
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_history(train_losses: list,
                             val_losses: list,
                             train_accs: list = None,
                             val_accs: list = None,
                             figsize: Tuple = (12, 4)):
        """
        Plot training history
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            train_accs: Training accuracies
            val_accs: Validation accuracies
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2 if train_accs is not None else 1, figsize=figsize)
        
        if train_accs is None:
            axes = [axes]
        
        # Loss plot
        axes[0].plot(train_losses, label='Train')
        axes[0].plot(val_losses, label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if train_accs is not None:
            axes[1].plot(train_accs, label='Train')
            axes[1].plot(val_accs, label='Validation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_metrics_bar(metrics: dict,
                        figsize: Tuple = (8, 5)):
        """
        Plot metrics as bar chart
        
        Args:
            metrics: Dictionary of metrics
            figsize: Figure size
        """
        names = list(metrics.keys())
        values = list(metrics.values())
        
        plt.figure(figsize=figsize)
        bars = plt.bar(names, values, color='steelblue', alpha=0.7)
        plt.ylabel('Score')
        plt.title('Model Metrics')
        plt.ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
