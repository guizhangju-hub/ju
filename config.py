"""Configuration file for TENG CNN model"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Configuration
SIGNAL_LENGTH = 1024  # Length of 1D sensor signal
SAMPLING_RATE = 1000  # Hz
NUM_CLASSES = 10  # Number of gesture classes

# Preprocessing Configuration
NORMALIZATION_METHOD = "standard"  # 'standard', 'minmax', 'robust'
NOISE_REDUCTION = True
NOISE_METHOD = "butterworth"  # 'butterworth', 'savitzky_golay'
AUGMENTATION_ENABLED = True

# Model Configuration
MODEL_NAME = "teng_cnn_1d"
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = NUM_CLASSES
KERNEL_SIZES = [3, 3, 3]
POOL_SIZES = [2, 2, 2]
FILTERS = [32, 64, 128]
DROPOUT_RATE = 0.5

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# Device
DEVICE = "cuda"  # 'cuda' or 'cpu'

# Random Seed
RANDOM_SEED = 42
