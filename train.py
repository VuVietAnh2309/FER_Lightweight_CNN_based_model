"""
Initial training script for PATTLite model.

This script performs the initial training phase with frozen backbone.
"""

import os
import argparse
import tensorflow as tf
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, create_directories
from dataset import prepare_data, get_data_summary
from model import create_pattlite_model, compile_model, get_model_summary
from utils.callbacks import get_training_callbacks
from utils.visualization import plot_training_history


def train(args):
    """
    Run initial training.

    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("PATTLITE MODEL - INITIAL TRAINING")
    print("="*60 + "\n")

    create_directories()

    print("Loading and preparing data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights = prepare_data(
        dataset_path=args.dataset_path if args.dataset_path else DATA_CONFIG['dataset_path'],
        dataset_name=args.dataset_name if args.dataset_name else DATA_CONFIG['dataset_name'],
        validation_split=DATA_CONFIG['validation_split'],
        random_state=DATA_CONFIG['random_state'],
        img_size=DATA_CONFIG['img_size'],
        compute_class_weights=TRAIN_CONFIG['use_class_weights']
    )

    get_data_summary(X_train, y_train, X_valid, y_valid, X_test, y_test)

    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60 + "\n")

    model = create_pattlite_model(
        num_classes=DATA_CONFIG['num_classes'],
        img_shape=DATA_CONFIG['img_shape'],
        mode='train',
        dropout=TRAIN_CONFIG['dropout']
    )

    model = compile_model(
        model,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        global_clipnorm=TRAIN_CONFIG['global_clipnorm'],
        loss=TRAIN_CONFIG['loss'],
        metrics=TRAIN_CONFIG['metrics']
    )

    get_model_summary(model)

    print("\n" + "="*60)
    print("PREPARING CALLBACKS")
    print("="*60 + "\n")

    callbacks = get_training_callbacks(TRAIN_CONFIG)

    for i, callback in enumerate(callbacks):
        print(f"{i+1}. {callback.__class__.__name__}")

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    print(f"Training parameters:")
    print(f"  Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"  Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Dropout: {TRAIN_CONFIG['dropout']}")
    print(f"  Using class weights: {TRAIN_CONFIG['use_class_weights']}")
    print()

    history = model.fit(
        X_train, y_train,
        epochs=TRAIN_CONFIG['epochs'],
        batch_size=TRAIN_CONFIG['batch_size'],
        validation_data=(X_valid, y_valid),
        verbose=1,
        class_weight=class_weights if TRAIN_CONFIG['use_class_weights'] else None,
        callbacks=callbacks
    )

    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60 + "\n")

    model_path = args.model_path if args.model_path else TRAIN_CONFIG['checkpoint_path']
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60 + "\n")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\n" + "="*60)
    print("PLOTTING TRAINING HISTORY")
    print("="*60 + "\n")

    plot_path = 'results/training_history.png'
    plot_training_history(history, save_path=plot_path, show=False)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print(f"Training history plot saved to: {plot_path}")
    print(f"Final test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nNext steps:")
    print("  1. Run fine-tuning: python finetune.py")
    print("  2. Evaluate model: python evaluate.py")
    print("  3. Make predictions: python predict.py --image <path_to_image>")
    print("="*60 + "\n")


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train PATTLite model for Facial Expression Recognition'
    )

    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help=f'Path to dataset (default: {DATA_CONFIG["dataset_path"]})'
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        default=None,
        help=f'Dataset folder name (default: {DATA_CONFIG["dataset_name"]})'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help=f'Path to save trained model (default: {TRAIN_CONFIG["checkpoint_path"]})'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of training epochs (default: {TRAIN_CONFIG["epochs"]})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size (default: {TRAIN_CONFIG["batch_size"]})'
    )

    args = parser.parse_args()

    if args.epochs is not None:
        TRAIN_CONFIG['epochs'] = args.epochs

    if args.batch_size is not None:
        TRAIN_CONFIG['batch_size'] = args.batch_size

    train(args)


if __name__ == '__main__':
    main()
