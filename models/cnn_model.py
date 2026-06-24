"""1D CNN model for TENG sensor data classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TENGCNN1D(nn.Module):
    """1D CNN model for TENG sensor signal classification"""
    
    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 10,
                 kernel_sizes: List[int] = None,
                 pool_sizes: List[int] = None,
                 filters: List[int] = None,
                 dropout_rate: float = 0.5):
        """
        Initialize 1D CNN model
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            kernel_sizes: Kernel sizes for each conv layer
            pool_sizes: Pool sizes for each pooling layer
            filters: Number of filters for each conv layer
            dropout_rate: Dropout probability
        """
        super(TENGCNN1D, self).__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if pool_sizes is None:
            pool_sizes = [2, 2, 2]
        if filters is None:
            filters = [32, 64, 128]
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = input_channels
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(filters, kernel_sizes, pool_sizes)
        ):
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                         padding=kernel_size//2)
            )
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            self.pool_layers.append(nn.MaxPool1d(kernel_size=pool_size))
            in_channels = out_channels
        
        # Calculate flattened dimension (approximate for 1024 length signal)
        flattened_dim = filters[-1] * (1024 // (2**len(pool_sizes)))
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, signal_length)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional blocks
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = pool(x)
            x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResNetBlock1D(nn.Module):
    """Residual block for 1D convolution"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ResNetBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                               padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential() if in_channels == out_channels else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        
        return out


class ResNetCNN1D(nn.Module):
    """ResNet-based 1D CNN for TENG sensor data"""
    
    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 10,
                 num_blocks: List[int] = None,
                 filters: List[int] = None):
        """
        Initialize ResNet-based 1D CNN
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            num_blocks: Number of residual blocks for each stage
            filters: Number of filters for each stage
        """
        super(ResNetCNN1D, self).__init__()
        
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        if filters is None:
            filters = [64, 128, 256, 512]
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, filters[0], kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(filters[0])
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.stages = nn.ModuleList()
        in_channels = filters[0]
        for stage, (num_block, out_channels) in enumerate(zip(num_blocks, filters)):
            blocks = nn.ModuleList()
            for i in range(num_block):
                ch_in = in_channels if i == 0 else out_channels
                blocks.append(ResNetBlock1D(ch_in, out_channels))
            self.stages.append(blocks)
            in_channels = out_channels
        
        # Global average pooling and FC layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters[-1], num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, signal_length)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Residual stages
        for stage in self.stages:
            for block in stage:
                x = block(x)
            x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
