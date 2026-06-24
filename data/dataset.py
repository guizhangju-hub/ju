"""Dataset class for TENG sensor data"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Callable
from pathlib import Path
import pickle


class TENGDataset(Dataset):
    """Dataset for TENG sensor data"""
    
    def __init__(self,
                 signals: np.ndarray,
                 labels: np.ndarray,
                 transform: Optional[Callable] = None):
        """
        Initialize dataset
        
        Args:
            signals: Array of signals with shape (n_samples, signal_length)
            labels: Array of labels with shape (n_samples,)
            transform: Optional augmentation/preprocessing function
        """
        self.signals = signals
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (signal, label) as tensors
        """
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.transform:
            signal = self.transform(signal)
        
        # Convert to tensor and add channel dimension
        signal = torch.FloatTensor(signal).unsqueeze(0)  # (1, signal_length)
        label = torch.LongTensor([label]).squeeze()
        
        return signal, label
    
    @classmethod
    def from_file(cls, filepath: str, transform: Optional[Callable] = None):
        """
        Load dataset from file
        
        Args:
            filepath: Path to dataset file
            transform: Optional transformation
            
        Returns:
            TENGDataset instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        signals = data['signals']
        labels = data['labels']
        
        return cls(signals, labels, transform)
    
    def save(self, filepath: str):
        """
        Save dataset to file
        
        Args:
            filepath: Path to save dataset
        """
        data = {
            'signals': self.signals,
            'labels': self.labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)


def create_dummy_dataset(n_samples: int = 1000,
                        signal_length: int = 1024,
                        num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dummy TENG sensor dataset for testing
    
    Args:
        n_samples: Number of samples
        signal_length: Length of each signal
        num_classes: Number of classes
        
    Returns:
        Tuple of (signals, labels)
    """
    signals = np.random.randn(n_samples, signal_length).astype(np.float32)
    labels = np.random.randint(0, num_classes, n_samples)
    
    return signals, labels


def get_data_loaders(train_signals: np.ndarray,
                    train_labels: np.ndarray,
                    val_signals: np.ndarray,
                    val_labels: np.ndarray,
                    test_signals: np.ndarray,
                    test_labels: np.ndarray,
                    batch_size: int = 32,
                    num_workers: int = 4,
                    transform: Optional[Callable] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        train_signals: Training signals
        train_labels: Training labels
        val_signals: Validation signals
        val_labels: Validation labels
        test_signals: Test signals
        test_labels: Test labels
        batch_size: Batch size
        num_workers: Number of workers for data loading
        transform: Optional augmentation function
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = TENGDataset(train_signals, train_labels, transform=transform)
    val_dataset = TENGDataset(val_signals, val_labels, transform=None)
    test_dataset = TENGDataset(test_signals, test_labels, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
