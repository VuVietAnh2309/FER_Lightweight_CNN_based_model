"""
Custom callbacks for model training.

This module provides utility functions to create and configure
callbacks for training and fine-tuning.
"""

import os
import datetime
import tensorflow as tf
from tensorflow import keras


def get_early_stopping_callback(monitor='val_accuracy', patience=5,
                                min_delta=0.003, restore_best_weights=True,
                                verbose=1):
    """
    Create an EarlyStopping callback.

    Args:
        monitor (str): Metric to monitor
        patience (int): Number of epochs with no improvement to wait
        min_delta (float): Minimum change to qualify as improvement
        restore_best_weights (bool): Whether to restore best weights
        verbose (int): Verbosity mode

    Returns:
        tf.keras.callbacks.EarlyStopping: Early stopping callback
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        verbose=verbose
    )


def get_reduce_lr_callback(monitor='val_accuracy', patience=3,
                           min_delta=0.003, min_lr=1e-6,
                           factor=0.1, verbose=1):
    """
    Create a ReduceLROnPlateau callback.

    Args:
        monitor (str): Metric to monitor
        patience (int): Number of epochs with no improvement to wait
        min_delta (float): Minimum change to qualify as improvement
        min_lr (float): Minimum learning rate
        factor (float): Factor by which to reduce learning rate
        verbose (int): Verbosity mode

    Returns:
        tf.keras.callbacks.ReduceLROnPlateau: Learning rate reduction callback
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        patience=patience,
        verbose=verbose,
        min_delta=min_delta,
        min_lr=min_lr,
        factor=factor
    )


def get_model_checkpoint_callback(filepath, monitor='val_accuracy',
                                  save_best_only=True, save_weights_only=False,
                                  verbose=1):
    """
    Create a ModelCheckpoint callback.

    Args:
        filepath (str): Path to save the model
        monitor (str): Metric to monitor
        save_best_only (bool): Whether to save only the best model
        save_weights_only (bool): Whether to save only weights
        verbose (int): Verbosity mode

    Returns:
        tf.keras.callbacks.ModelCheckpoint: Model checkpoint callback
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        verbose=verbose
    )


def get_tensorboard_callback(log_dir='logs/fit/', histogram_freq=1):
    """
    Create a TensorBoard callback.

    Args:
        log_dir (str): Directory for TensorBoard logs
        histogram_freq (int): Frequency for writing histograms

    Returns:
        tf.keras.callbacks.TensorBoard: TensorBoard callback
    """
    log_dir_with_time = os.path.join(
        log_dir,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    os.makedirs(log_dir_with_time, exist_ok=True)

    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir_with_time,
        histogram_freq=histogram_freq
    )


def get_lr_scheduler_callback(initial_learning_rate, decay_steps, decay_rate):
    """
    Create a LearningRateScheduler callback with InverseTimeDecay.

    Args:
        initial_learning_rate (float): Initial learning rate
        decay_steps (float): Decay steps for learning rate schedule
        decay_rate (float): Decay rate for learning rate schedule

    Returns:
        tf.keras.callbacks.LearningRateScheduler: Learning rate scheduler callback
    """
    scheduler = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )

    return tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)


def get_training_callbacks(config):
    """
    Get all callbacks for initial training.

    Args:
        config (dict): Training configuration dictionary

    Returns:
        list: List of callbacks for training
    """
    callbacks = [
        get_early_stopping_callback(
            monitor=config['early_stopping_monitor'],
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta'],
            restore_best_weights=config['restore_best_weights']
        ),
        get_reduce_lr_callback(
            monitor=config['reduce_lr_monitor'],
            patience=config['reduce_lr_patience'],
            min_delta=config['reduce_lr_min_delta'],
            min_lr=config['reduce_lr_min_lr']
        )
    ]

    if 'checkpoint_path' in config:
        callbacks.append(
            get_model_checkpoint_callback(
                filepath=config['checkpoint_path'],
                save_best_only=config.get('save_best_only', True)
            )
        )

    return callbacks


def get_finetuning_callbacks(config):
    """
    Get all callbacks for fine-tuning.

    Args:
        config (dict): Fine-tuning configuration dictionary

    Returns:
        list: List of callbacks for fine-tuning
    """
    callbacks = [
        get_early_stopping_callback(
            monitor=config['early_stopping_monitor'],
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta'],
            restore_best_weights=config['restore_best_weights']
        ),
        get_lr_scheduler_callback(
            initial_learning_rate=config['learning_rate'],
            decay_steps=config['lr_decay_steps'],
            decay_rate=config['lr_decay_rate']
        )
    ]

    if 'checkpoint_path' in config:
        callbacks.append(
            get_model_checkpoint_callback(
                filepath=config['checkpoint_path'],
                save_best_only=config.get('save_best_only', True)
            )
        )

    if config.get('use_tensorboard', False):
        callbacks.append(
            get_tensorboard_callback(
                log_dir=config['log_dir']
            )
        )

    return callbacks


if __name__ == '__main__':
    from config import TRAIN_CONFIG, FINETUNE_CONFIG

    print("Testing callback creation...\n")

    print("="*60)
    print("TRAINING CALLBACKS")
    print("="*60)
    training_callbacks = get_training_callbacks(TRAIN_CONFIG)
    for i, callback in enumerate(training_callbacks):
        print(f"{i+1}. {callback.__class__.__name__}")

    print("\n" + "="*60)
    print("FINE-TUNING CALLBACKS")
    print("="*60)
    finetuning_callbacks = get_finetuning_callbacks(FINETUNE_CONFIG)
    for i, callback in enumerate(finetuning_callbacks):
        print(f"{i+1}. {callback.__class__.__name__}")

    print("\nCallback creation test completed successfully!")
