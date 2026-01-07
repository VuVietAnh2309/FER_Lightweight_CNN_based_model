"""
Configuration file for PATTLite Facial Expression Recognition.

This module contains all hyperparameters and configuration settings
for data loading, model architecture, training, and fine-tuning.
"""

import os


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    # Dataset paths
    'dataset_path': '/kaggle/input',  # Update this to your dataset path
    'dataset_name': 'raf-db-dataset/DATASET',

    # Data splits
    'splits': ['train', 'test'],
    'validation_split': 0.1,
    'random_state': 42,

    # Image settings
    'img_size': 120,
    'img_shape': (120, 120, 3),
    'resize_size': 224,  # For MobileNet input

    # Class information
    'num_classes': 8,
    'class_names': ['1', '2', '3', '4', '5', '6', '7'],
}


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Backbone settings
    'backbone': 'mobilenet',
    'backbone_weights': 'imagenet',
    'backbone_input_shape': (224, 224, 3),
    'include_top': False,

    # Feature extraction settings
    'base_model_output_layer': -29,  # MobileNet layer to extract features from
    'unfreeze_layers': 59,  # Number of layers to unfreeze for fine-tuning

    # Patch extraction settings
    'patch_filters': 256,
    'patch_kernel_sizes': [4, 2, 1],
    'patch_strides': [4, 2, 1],

    # Classification head settings
    'pre_classification_units': 32,
    'use_attention': True,
}


# ============================================================================
# TRAINING CONFIGURATION (Initial Training)
# ============================================================================

TRAIN_CONFIG = {
    # Training parameters
    'epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-3,
    'dropout': 0.1,
    'use_class_weights': True,

    # Optimizer settings
    'optimizer': 'adam',
    'global_clipnorm': 3.0,

    # Loss and metrics
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy'],

    # Callbacks settings
    'early_stopping_patience': 5,
    'early_stopping_min_delta': 0.003,
    'early_stopping_monitor': 'val_accuracy',
    'restore_best_weights': True,

    'reduce_lr_patience': 3,
    'reduce_lr_monitor': 'val_accuracy',
    'reduce_lr_min_delta': 0.003,
    'reduce_lr_min_lr': 1e-6,

    # Model checkpoint
    'checkpoint_path': 'checkpoints/model_initial.h5',
    'save_best_only': True,
}


# ============================================================================
# FINE-TUNING CONFIGURATION
# ============================================================================

FINETUNE_CONFIG = {
    # Training parameters
    'epochs': 65,
    'batch_size': 8,
    'learning_rate': 1e-5,
    'dropout': 0.2,
    'spatial_dropout': 0.2,

    # Optimizer settings
    'optimizer': 'adam',
    'global_clipnorm': 3.0,

    # Loss and metrics
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy'],

    # Callbacks settings
    'early_stopping_patience': 20,
    'early_stopping_min_delta': 0.003,
    'early_stopping_monitor': 'accuracy',
    'restore_best_weights': True,

    # Learning rate scheduler
    'lr_decay_steps': 80.0,
    'lr_decay_rate': 1,

    # Model checkpoint
    'checkpoint_path': 'checkpoints/model_finetune.h5',
    'save_best_only': True,

    # TensorBoard
    'use_tensorboard': True,
    'log_dir': 'logs/fit/',
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVAL_CONFIG = {
    'model_path': 'checkpoints/model_finetune.h5',
    'batch_size': 32,
    'save_results': True,
    'results_dir': 'results/',
    'confusion_matrix_path': 'results/confusion_matrix.png',
    'classification_report_path': 'results/classification_report.txt',
}


# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

PREDICT_CONFIG = {
    'model_path': 'checkpoints/model_finetune.h5',
    'img_size': 120,
    'emotion_labels': {
        0: 'Class 1',
        1: 'Class 2',
        2: 'Class 3',
        3: 'Class 4',
        4: 'Class 5',
        5: 'Class 6',
        6: 'Class 7',
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """Create necessary directories for saving models, logs, and results."""
    directories = [
        'checkpoints',
        'logs',
        'logs/fit',
        'results',
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Created directories:")
    for directory in directories:
        print(f"  - {directory}")


def get_config(config_type='train'):
    """
    Get configuration dictionary by type.

    Args:
        config_type (str): Type of configuration to retrieve.
            Options: 'data', 'model', 'train', 'finetune', 'eval', 'predict'

    Returns:
        dict: Configuration dictionary
    """
    configs = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'train': TRAIN_CONFIG,
        'finetune': FINETUNE_CONFIG,
        'eval': EVAL_CONFIG,
        'predict': PREDICT_CONFIG,
    }

    if config_type not in configs:
        raise ValueError(f"Invalid config_type. Choose from: {list(configs.keys())}")

    return configs[config_type]


if __name__ == '__main__':
    # Create directories when running this file
    create_directories()

    # Print configuration summary
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)

    print(f"\nImage Size: {DATA_CONFIG['img_size']}")
    print(f"Number of Classes: {DATA_CONFIG['num_classes']}")
    print(f"Training Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"Fine-tuning Epochs: {FINETUNE_CONFIG['epochs']}")
    print(f"Batch Size: {TRAIN_CONFIG['batch_size']}")
    print(f"Initial Learning Rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"Fine-tuning Learning Rate: {FINETUNE_CONFIG['learning_rate']}")
    print("\n" + "="*60)
