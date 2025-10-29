"""
Training Configuration

Hyperparameters and settings for training the baseline ADL recognition model.
"""

import torch

class TrainingConfig:
    """Configuration for baseline model training"""
    
    # Model architecture
    INPUT_DIM = 114  # Number of input features
    HIDDEN_DIM = 128
    N_CLASSES = 5  # t1, t2, t3, t4, t5
    N_TCN_BLOCKS = 3
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4]
    DROPOUT = 0.3
    
    # Training hyperparameters
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    PATIENCE = 7  # Early stopping patience
    
    # Loss function (Focal Loss for class imbalance)
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = None  # Will be computed from class weights
    
    # Data augmentation
    USE_WEIGHTED_SAMPLER = True  # Balance classes during training
    
    # Optimizer
    OPTIMIZER = 'AdamW'
    WEIGHT_DECAY = 1e-4
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER = 'CosineAnnealingWarmRestarts'
    T_0 = 10  # First restart after 10 epochs
    T_MULT = 1
    ETA_MIN = 1e-6
    
    # Data split
    TRAIN_RATIO = 0.64
    VAL_RATIO = 0.16
    TEST_RATIO = 0.20
    RANDOM_SEED = 42
    
    # Sequence parameters
    SEQUENCE_LENGTH = 100  # T (time steps)
    SLIDING_WINDOW_STRIDE = 5
    
    # Checkpoint and logging
    CHECKPOINT_DIR = '../checkpoints'
    LOG_DIR = '../logs'
    SAVE_BEST_ONLY = True
    VERBOSE = 1  # 0: silent, 1: progress bar, 2: one line per epoch
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Paths
    DATA_PATH = '../data/processed/dataset.npz'
    EMBEDDING_PATH = '../data/embeddings/sensor_embeddings.npz'
    
    @classmethod
    def get_model_config(cls):
        """Get model configuration dict"""
        return {
            'input_dim': cls.INPUT_DIM,
            'hidden_dim': cls.HIDDEN_DIM,
            'n_classes': cls.N_CLASSES,
            'n_tcn_blocks': cls.N_TCN_BLOCKS,
            'kernel_size': cls.KERNEL_SIZE,
            'dilations': cls.DILATIONS,
            'dropout': cls.DROPOUT
        }
    
    @classmethod
    def get_optimizer_config(cls):
        """Get optimizer configuration dict"""
        return {
            'lr': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY
        }
    
    @classmethod
    def get_scheduler_config(cls):
        """Get scheduler configuration dict"""
        return {
            'T_0': cls.T_0,
            'T_mult': cls.T_MULT,
            'eta_min': cls.ETA_MIN
        }
    
    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print("=" * 80)
        print("Training Configuration")
        print("=" * 80)
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Early Stopping Patience: {cls.PATIENCE}")
        print(f"Hidden Dim: {cls.HIDDEN_DIM}")
        print(f"TCN Blocks: {cls.N_TCN_BLOCKS}")
        print(f"Dilations: {cls.DILATIONS}")
        print(f"Dropout: {cls.DROPOUT}")
        print(f"Focal Loss Gamma: {cls.FOCAL_GAMMA}")
        print(f"Weighted Sampler: {cls.USE_WEIGHTED_SAMPLER}")
        print(f"Scheduler: {cls.SCHEDULER if cls.USE_SCHEDULER else 'None'}")
        print("=" * 80)


# Alternative configurations for experimentation

class FastTrainConfig(TrainingConfig):
    """Faster training for testing"""
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    PATIENCE = 3


class LargeModelConfig(TrainingConfig):
    """Larger model for higher accuracy"""
    HIDDEN_DIM = 256
    N_TCN_BLOCKS = 4
    DILATIONS = [1, 2, 4, 8]
    DROPOUT = 0.4


class SmallModelConfig(TrainingConfig):
    """Smaller model for efficiency"""
    HIDDEN_DIM = 64
    N_TCN_BLOCKS = 2
    DILATIONS = [1, 2]
    DROPOUT = 0.2


if __name__ == '__main__':
    # Test configuration
    TrainingConfig.print_config()
    
    print("\n" + "=" * 80)
    print("Model Config:")
    print("=" * 80)
    for k, v in TrainingConfig.get_model_config().items():
        print(f"  {k}: {v}")
