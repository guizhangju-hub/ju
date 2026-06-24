"""Normalization module for sensor data"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Optional


class DataNormalizer:
    """Normalize sensor data using various methods"""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize normalizer
        
        Args:
            method: 'standard', 'minmax', or 'robust'
        """
        self.method = method
        self.scaler = None
        self._init_scaler()
    
    def _init_scaler(self):
        """Initialize scaler based on method"""
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def fit(self, data: np.ndarray) -> 'DataNormalizer':
        """
        Fit normalizer on data
        
        Args:
            data: Training data with shape (n_samples, signal_length)
            
        Returns:
            Self
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        self.scaler.fit(data)
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted normalizer
        
        Args:
            data: Data to transform
            
        Returns:
            Normalized data
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        normalized = self.scaler.transform(data)
        
        if normalized.shape[1] == 1:
            normalized = normalized.flatten()
        
        return normalized
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform data
        
        Args:
            data: Data to fit and transform
            
        Returns:
            Normalized data
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data
        
        Args:
            data: Normalized data
            
        Returns:
            Original scale data
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        original = self.scaler.inverse_transform(data)
        
        if original.shape[1] == 1:
            original = original.flatten()
        
        return original
    
    @staticmethod
    def manual_standardization(data: np.ndarray,
                              mean: Optional[np.ndarray] = None,
                              std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Manual standardization (z-score normalization)
        
        Args:
            data: Input data
            mean: Mean values (compute if None)
            std: Standard deviation values (compute if None)
            
        Returns:
            Standardized data
        """
        if mean is None:
            mean = np.mean(data, axis=0)
        if std is None:
            std = np.std(data, axis=0)
        
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        return (data - mean) / std
    
    @staticmethod
    def manual_minmax(data: np.ndarray,
                     data_min: Optional[np.ndarray] = None,
                     data_max: Optional[np.ndarray] = None,
                     feature_range: tuple = (0, 1)) -> np.ndarray:
        """
        Manual min-max scaling
        
        Args:
            data: Input data
            data_min: Minimum values (compute if None)
            data_max: Maximum values (compute if None)
            feature_range: Output range (default: 0 to 1)
            
        Returns:
            Scaled data
        """
        if data_min is None:
            data_min = np.min(data, axis=0)
        if data_max is None:
            data_max = np.max(data, axis=0)
        
        range_val = data_max - data_min
        range_val = np.where(range_val == 0, 1, range_val)
        
        scaled = (data - data_min) / range_val
        scaled = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        return scaled
