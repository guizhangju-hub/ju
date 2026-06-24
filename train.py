"""Training script for TENG CNN model"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, MODEL_DIR, LOG_DIR, RANDOM_SEED,
    SIGNAL_LENGTH, NUM_CLASSES, INPUT_CHANNELS, KERNEL_SIZES,
    POOL_SIZES, FILTERS, DROPOUT_RATE
)
from models.cnn_model import TENGCNN1D, ResNetCNN1D
from data.dataset import get_data_loaders, create_dummy_dataset, TENGDataset
from preprocessing.augmentation import SignalAugmentation
from utils.metrics import AverageMeter, get_accuracy


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Training wrapper for CNN model"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: str = 'cpu',
                 lr: float = 0.001,
                 weight_decay: float = 1e-4):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.patience_counter = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for signals, labels in pbar:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                acc = get_accuracy(outputs, labels)
                loss_meter.update(loss.item(), signals.size(0))
                acc_meter.update(acc, signals.size(0))
                
                pbar.set_postfix({'loss': loss_meter.avg, 'acc': acc_meter.avg})
        
        return loss_meter.avg, acc_meter.avg
    
    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validating") as pbar:
                for signals, labels in pbar:
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)
                    
                    acc = get_accuracy(outputs, labels)
                    loss_meter.update(loss.item(), signals.size(0))
                    acc_meter.update(acc, signals.size(0))
                    
                    pbar.set_postfix({'loss': loss_meter.avg, 'acc': acc_meter.avg})
        
        return loss_meter.avg, acc_meter.avg
    
    def test(self) -> Tuple[float, float]:
        """
        Test model
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        with torch.no_grad():
            with tqdm(self.test_loader, desc="Testing") as pbar:
                for signals, labels in pbar:
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)
                    
                    acc = get_accuracy(outputs, labels)
                    loss_meter.update(loss.item(), signals.size(0))
                    acc_meter.update(acc, signals.size(0))
                    
                    pbar.set_postfix({'loss': loss_meter.avg, 'acc': acc_meter.avg})
        
        return loss_meter.avg, acc_meter.avg
    
    def train(self, num_epochs: int, patience: int = 15):
        """
        Train model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            patience: Early stopping patience
        """
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, "best_model.pt")
                print(f"Best model saved with Val Acc: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
    
    def save_checkpoint(self, epoch: int, filename: str):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            filename: Filename for checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }
        torch.save(checkpoint, MODEL_DIR / filename)
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint
        
        Args:
            filename: Filename of checkpoint
        """
        checkpoint = torch.load(MODEL_DIR / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']


def main():
    """Main training function"""
    set_seed(RANDOM_SEED)
    
    print("Loading data...")
    # Create dummy dataset (replace with your actual data loading)
    signals, labels = create_dummy_dataset(
        n_samples=1000,
        signal_length=SIGNAL_LENGTH,
        num_classes=NUM_CLASSES
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_signals, test_signals, train_labels, test_labels = train_test_split(
        signals, labels, test_size=0.2, random_state=RANDOM_SEED
    )
    train_signals, val_signals, train_labels, val_labels = train_test_split(
        train_signals, train_labels, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Create data loaders with augmentation
    def augment_fn(signal):
        return SignalAugmentation.compose_augmentations(signal)
    
    train_loader, val_loader, test_loader = get_data_loaders(
        train_signals, train_labels,
        val_signals, val_labels,
        test_signals, test_labels,
        batch_size=BATCH_SIZE,
        num_workers=0,
        transform=augment_fn
    )
    
    print("Building model...")
    model = TENGCNN1D(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        kernel_sizes=KERNEL_SIZES,
        pool_sizes=POOL_SIZES,
        filters=FILTERS,
        dropout_rate=DROPOUT_RATE
    )
    
    # Move model to device
    device = DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    print("Starting training...")
    trainer.train(num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE)
    
    print("\nTesting...")
    test_loss, test_acc = trainer.test()
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save final model
    trainer.save_checkpoint(NUM_EPOCHS - 1, "final_model.pt")
    print(f"\nTraining complete! Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
