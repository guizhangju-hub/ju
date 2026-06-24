"""Signal processing module for TENG sensor data"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional


class SignalProcessor:
    """Process 1D TENG sensor signals"""
    
    def __init__(self, sampling_rate: int = 1000, signal_length: int = 1024):
        """
        Initialize signal processor
        
        Args:
            sampling_rate: Sampling rate in Hz
            signal_length: Expected signal length
        """
        self.sampling_rate = sampling_rate
        self.signal_length = signal_length
        self.nyquist = sampling_rate / 2
    
    def apply_butterworth_filter(self, signal_data: np.ndarray, 
                                 cutoff_freq: float = 100,
                                 order: int = 4,
                                 filter_type: str = 'low') -> np.ndarray:
        """
        Apply Butterworth filter for noise reduction
        
        Args:
            signal_data: Input signal
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order
            filter_type: 'low', 'high', or 'band'
            
        Returns:
            Filtered signal
        """
        normalized_cutoff = cutoff_freq / self.nyquist
        
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
        
        b, a = signal.butter(order, normalized_cutoff, btype=filter_type)
        filtered = signal.filtfilt(b, a, signal_data)
        
        return filtered
    
    def apply_savitzky_golay_filter(self, signal_data: np.ndarray,
                                    window_length: int = 11,
                                    polyorder: int = 3) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for smoothing
        
        Args:
            signal_data: Input signal
            window_length: Window length (must be odd)
            polyorder: Polynomial order
            
        Returns:
            Filtered signal
        """
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length > len(signal_data):
            window_length = len(signal_data) if len(signal_data) % 2 == 1 else len(signal_data) - 1
        
        if window_length < polyorder + 2:
            polyorder = max(1, window_length - 2)
        
        filtered = signal.savgol_filter(signal_data, window_length, polyorder)
        
        return filtered
    
    def remove_outliers(self, signal_data: np.ndarray,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> np.ndarray:
        """
        Remove outliers from signal
        
        Args:
            signal_data: Input signal
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or z-score threshold
            
        Returns:
            Signal with outliers removed/replaced
        """
        cleaned = signal_data.copy()
        
        if method == 'iqr':
            Q1 = np.percentile(signal_data, 25)
            Q3 = np.percentile(signal_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (signal_data < lower_bound) | (signal_data > upper_bound)
            cleaned[outlier_mask] = np.median(signal_data)
        
        elif method == 'zscore':
            z_scores = np.abs((signal_data - np.mean(signal_data)) / np.std(signal_data))
            outlier_mask = z_scores > threshold
            cleaned[outlier_mask] = np.median(signal_data)
        
        return cleaned
    
    def compute_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of signal
        
        Args:
            signal_data: Input signal
            
        Returns:
            Frequency and magnitude arrays
        """
        fft = np.fft.fft(signal_data)
        magnitude = np.abs(fft)
        freq = np.fft.fftfreq(len(signal_data), 1/self.sampling_rate)
        
        # Return only positive frequencies
        positive_idx = freq >= 0
        return freq[positive_idx], magnitude[positive_idx]
    
    def extract_statistical_features(self, signal_data: np.ndarray) -> dict:
        """
        Extract statistical features from signal
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary of statistical features
        """
        features = {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'min': np.min(signal_data),
            'max': np.max(signal_data),
            'median': np.median(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'skewness': float(signal.skew(signal_data)),
            'kurtosis': float(signal.kurtosis(signal_data)),
            'energy': np.sum(signal_data**2),
            'peak_to_peak': np.max(signal_data) - np.min(signal_data),
        }
        return features
