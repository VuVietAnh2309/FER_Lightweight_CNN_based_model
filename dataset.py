"""
Dataset loading and preprocessing for RAF-DB Facial Expression Recognition.

This module provides functions to load and prepare the RAF-DB dataset
for training and evaluation.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from config import DATA_CONFIG


def load_data(path, dataset_name, splits=['train', 'test'], img_size=120):
    """
    Load RAF-DB dataset from directory structure.

    Args:
        path (str): Root path to the dataset
        dataset_name (str): Name of the dataset folder
        splits (list): List of splits to load (e.g., ['train', 'test'])
        img_size (int): Target image size for resizing

    Returns:
        tuple: (X, y) dictionaries containing images and labels for each split
            X: dict of numpy arrays with shape (num_samples, img_size, img_size, 3)
            y: dict of numpy arrays with shape (num_samples,)
    """
    X, y = {}, {}

    class_names = DATA_CONFIG['class_names']

    for split in splits:
        dataset_path = os.path.join(path, dataset_name, split)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        X[split], y[split] = [], []

        print(f"\nLoading {split} split from {dataset_path}")

        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)

            if not os.path.isdir(class_path):
                continue

            if class_name not in class_names:
                print(f"Warning: Skipping unknown class '{class_name}'")
                continue

            class_idx = class_names.index(class_name)

            for sample_file in os.listdir(class_path):
                sample_path = os.path.join(class_path, sample_file)

                try:
                    image = cv2.imread(sample_path, cv2.IMREAD_COLOR)

                    if image is None:
                        print(f"Warning: Failed to load image {sample_path}")
                        continue

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (img_size, img_size))

                    X[split].append(image)
                    y[split].append(class_idx)

                except Exception as e:
                    print(f"Error loading {sample_path}: {str(e)}")
                    continue

        X[split] = np.array(X[split])
        y[split] = np.array(y[split])

        print(f"Loaded {len(X[split])} samples for {split} split")
        print(f"  - Shape: {X[split].shape}")
        print(f"  - Labels shape: {y[split].shape}")
        print(f"  - Label distribution: {np.bincount(y[split])}")

    return X, y


def prepare_data(dataset_path=None, dataset_name=None, validation_split=0.1,
                 random_state=42, img_size=120, compute_class_weights=True):
    """
    Load and prepare data for training.

    This function:
    1. Loads train and test data
    2. Splits training data into train and validation sets
    3. Shuffles the training data
    4. Computes class weights if requested

    Args:
        dataset_path (str): Root path to dataset. If None, uses config
        dataset_name (str): Dataset folder name. If None, uses config
        validation_split (float): Fraction of training data for validation
        random_state (int): Random seed for reproducibility
        img_size (int): Image size for resizing
        compute_class_weights (bool): Whether to compute class weights

    Returns:
        tuple: (X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights)
    """
    if dataset_path is None:
        dataset_path = DATA_CONFIG['dataset_path']
    if dataset_name is None:
        dataset_name = DATA_CONFIG['dataset_name']

    print("="*60)
    print("LOADING DATASET")
    print("="*60)

    X, y = load_data(dataset_path, dataset_name, splits=['train', 'test'], img_size=img_size)

    X_train = X['train']
    y_train = y['train']
    X_test = X['test']
    y_test = y['test']

    print(f"\n{'='*60}")
    print("CREATING VALIDATION SPLIT")
    print("="*60)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=random_state
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_valid.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print(f"\n{'='*60}")
    print("SHUFFLING TRAINING DATA")
    print("="*60)

    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

    print(f"Train samples shape: {X_train.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Validation samples shape: {X_valid.shape}")
    print(f"Validation labels shape: {y_valid.shape}")
    print(f"Test samples shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    class_weights = None
    if compute_class_weights:
        print(f"\n{'='*60}")
        print("COMPUTING CLASS WEIGHTS")
        print("="*60)

        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights_array))

        print("Class weights:")
        for class_idx, weight in class_weights.items():
            print(f"  Class {class_idx}: {weight:.4f}")

    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE")
    print("="*60 + "\n")

    return X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights


def get_data_summary(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Print a summary of the dataset.

    Args:
        X_train, y_train: Training data and labels
        X_valid, y_valid: Validation data and labels
        X_test, y_test: Test data and labels
    """
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    splits = {
        'Train': (X_train, y_train),
        'Validation': (X_valid, y_valid),
        'Test': (X_test, y_test)
    }

    for split_name, (X, y) in splits.items():
        print(f"\n{split_name} Set:")
        print(f"  - Samples: {len(X)}")
        print(f"  - Shape: {X.shape}")
        print(f"  - Data type: {X.dtype}")
        print(f"  - Value range: [{X.min()}, {X.max()}]")
        print(f"  - Labels: {y.shape}")
        print(f"  - Unique labels: {np.unique(y)}")
        print(f"  - Label distribution: {np.bincount(y)}")

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    print("Testing dataset loading...\n")

    X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights = prepare_data(
        validation_split=DATA_CONFIG['validation_split'],
        random_state=DATA_CONFIG['random_state'],
        img_size=DATA_CONFIG['img_size'],
        compute_class_weights=True
    )

    get_data_summary(X_train, y_train, X_valid, y_valid, X_test, y_test)

    print("Dataset loading test completed successfully!")
