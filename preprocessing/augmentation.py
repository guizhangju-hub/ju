"""Data augmentation module for sensor signals"""

import numpy as np
from typing import Tuple


class SignalAugmentation:
    """Augment 1D sensor signals for better model generalization"""
    
    @staticmethod
    def add_gaussian_noise(signal_data: np.ndarray, 
                          noise_level: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to signal
        
        Args:
            signal_data: Input signal
            noise_level: Standard deviation of noise
            
        Returns:
            Signal with added noise
        """
        noise = np.random.normal(0, noise_level, len(signal_data))
        return signal_data + noise
    
    @staticmethod
    def time_shift(signal_data: np.ndarray,
                   max_shift: int = 10) -> np.ndarray:
        """
        Randomly shift signal in time domain
        
        Args:
            signal_data: Input signal
            max_shift: Maximum shift amount in samples
            
        Returns:
            Time-shifted signal
        """
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal_data, shift)
    
    @staticmethod
    def time_stretch(signal_data: np.ndarray,
                     stretch_factor: float = 0.1) -> np.ndarray:
        """
        Randomly stretch or compress signal in time
        
        Args:
            signal_data: Input signal
            stretch_factor: Maximum stretch factor (0.1 = ±10%)
            
        Returns:
            Time-stretched signal
        """
        factor = 1 + np.random.uniform(-stretch_factor, stretch_factor)
        original_length = len(signal_data)
        
        # Resample using interpolation
        indices = np.linspace(0, original_length - 1, int(original_length * factor))
        stretched = np.interp(indices, np.arange(original_length), signal_data)
        
        # Pad or truncate to original length
        if len(stretched) > original_length:
            stretched = stretched[:original_length]
        elif len(stretched) < original_length:
            stretched = np.pad(stretched, (0, original_length - len(stretched)), mode='edge')
        
        return stretched
    
    @staticmethod
    def amplitude_scale(signal_data: np.ndarray,
                       scale_factor: float = 0.2) -> np.ndarray:
        """
        Randomly scale amplitude of signal
        
        Args:
            signal_data: Input signal
            scale_factor: Maximum scale factor (0.2 = 0.8x to 1.2x)
            
        Returns:
            Amplitude-scaled signal
        """
        factor = 1 + np.random.uniform(-scale_factor, scale_factor)
        return signal_data * factor
    
    @staticmethod
    def window_warp(signal_data: np.ndarray,
                    num_windows: int = 4,
                    max_warp: float = 0.2) -> np.ndarray:
        """
        Apply random warping to signal windows
        
        Args:
            signal_data: Input signal
            num_windows: Number of windows to divide signal into
            max_warp: Maximum warp ratio
            
        Returns:
            Window-warped signal
        """
        warped = signal_data.copy()
        window_size = len(signal_data) // num_windows
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < num_windows - 1 else len(signal_data)
            
            warp_factor = 1 + np.random.uniform(-max_warp, max_warp)
            original_length = end_idx - start_idx
            warped_length = int(original_length * warp_factor)
            
            window = signal_data[start_idx:end_idx]
            indices = np.linspace(0, original_length - 1, warped_length)
            warped_window = np.interp(indices, np.arange(original_length), window)
            
            # Adjust to original length
            if len(warped_window) > original_length:
                warped_window = warped_window[:original_length]
            elif len(warped_window) < original_length:
                warped_window = np.pad(warped_window, (0, original_length - len(warped_window)), mode='edge')
            
            warped[start_idx:end_idx] = warped_window
        
        return warped
    
    @staticmethod
    def magnitude_warp(signal_data: np.ndarray,
                      num_knots: int = 4) -> np.ndarray:
        """
        Apply random magnitude warping
        
        Args:
            signal_data: Input signal
            num_knots: Number of knots for warping
            
        Returns:
            Magnitude-warped signal
        """
        # Generate random knots
        knot_indices = np.sort(np.random.choice(len(signal_data), num_knots, replace=False))
        knot_values = np.random.uniform(0.5, 1.5, num_knots)
        
        # Add start and end points
        knot_indices = np.insert(knot_indices, 0, 0)
        knot_indices = np.append(knot_indices, len(signal_data) - 1)
        knot_values = np.insert(knot_values, 0, 1.0)
        knot_values = np.append(knot_values, 1.0)
        
        # Interpolate warping factors
        warp_factors = np.interp(np.arange(len(signal_data)), knot_indices, knot_values)
        
        return signal_data * warp_factors
    
    @staticmethod
    def compose_augmentations(signal_data: np.ndarray,
                             augmentations: list = None,
                             probabilities: list = None) -> np.ndarray:
        """
        Compose multiple augmentations
        
        Args:
            signal_data: Input signal
            augmentations: List of augmentation methods
            probabilities: Probability of applying each augmentation
            
        Returns:
            Augmented signal
        """
        if augmentations is None:
            augmentations = [
                SignalAugmentation.add_gaussian_noise,
                SignalAugmentation.time_shift,
                SignalAugmentation.amplitude_scale
            ]
        
        if probabilities is None:
            probabilities = [0.3, 0.3, 0.3]
        
        augmented = signal_data.copy()
        
        for aug, prob in zip(augmentations, probabilities):
            if np.random.random() < prob:
                augmented = aug(augmented)
        
        return augmented
