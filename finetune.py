"""
Fine-tuning script for PATTLite model.

This script performs fine-tuning by unfreezing backbone layers.
"""

import os
import argparse
import tensorflow as tf
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, FINETUNE_CONFIG, create_directories
from dataset import prepare_data, get_data_summary
from model import create_pattlite_model, unfreeze_base_model, compile_model, get_model_summary
from utils.callbacks import get_finetuning_callbacks
from utils.visualization import plot_training_history


def finetune(args):
    """
    Run fine-tuning.

    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("PATTLITE MODEL - FINE-TUNING")
    print("="*60 + "\n")

    create_directories()

    print("Loading and preparing data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_weights = prepare_data(
        dataset_path=args.dataset_path if args.dataset_path else DATA_CONFIG['dataset_path'],
        dataset_name=args.dataset_name if args.dataset_name else DATA_CONFIG['dataset_name'],
        validation_split=DATA_CONFIG['validation_split'],
        random_state=DATA_CONFIG['random_state'],
        img_size=DATA_CONFIG['img_size'],
        compute_class_weights=False
    )

    get_data_summary(X_train, y_train, X_valid, y_valid, X_test, y_test)

    print("\n" + "="*60)
    print("BUILDING MODEL FOR FINE-TUNING")
    print("="*60 + "\n")

    model = create_pattlite_model(
        num_classes=DATA_CONFIG['num_classes'],
        img_shape=DATA_CONFIG['img_shape'],
        mode='finetune',
        dropout=FINETUNE_CONFIG['dropout'],
        spatial_dropout=FINETUNE_CONFIG['spatial_dropout']
    )

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"\nLoading weights from: {args.pretrained_model}")
        try:
            pretrained = tf.keras.models.load_model(args.pretrained_model)

            for layer in model.layers:
                if layer.name in [l.name for l in pretrained.layers]:
                    try:
                        pretrained_layer = pretrained.get_layer(layer.name)
                        layer.set_weights(pretrained_layer.get_weights())
                    except:
                        pass

            print("Successfully loaded pretrained weights")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Starting fine-tuning from scratch...")

    print("\n" + "="*60)
    print("UNFREEZING BASE MODEL LAYERS")
    print("="*60)

    model = unfreeze_base_model(
        model,
        num_layers_to_unfreeze=MODEL_CONFIG['unfreeze_layers']
    )

    model = compile_model(
        model,
        learning_rate=FINETUNE_CONFIG['learning_rate'],
        global_clipnorm=FINETUNE_CONFIG['global_clipnorm'],
        loss=FINETUNE_CONFIG['loss'],
        metrics=FINETUNE_CONFIG['metrics']
    )

    get_model_summary(model)

    print("\n" + "="*60)
    print("PREPARING CALLBACKS")
    print("="*60 + "\n")

    callbacks = get_finetuning_callbacks(FINETUNE_CONFIG)

    for i, callback in enumerate(callbacks):
        print(f"{i+1}. {callback.__class__.__name__}")

    print("\n" + "="*60)
    print("STARTING FINE-TUNING")
    print("="*60 + "\n")

    print(f"Fine-tuning parameters:")
    print(f"  Epochs: {FINETUNE_CONFIG['epochs']}")
    print(f"  Batch size: {FINETUNE_CONFIG['batch_size']}")
    print(f"  Learning rate: {FINETUNE_CONFIG['learning_rate']}")
    print(f"  Dropout: {FINETUNE_CONFIG['dropout']}")
    print(f"  Spatial dropout: {FINETUNE_CONFIG['spatial_dropout']}")
    print(f"  Layers to unfreeze: {MODEL_CONFIG['unfreeze_layers']}")
    print()

    initial_epoch = 0
    if args.continue_from_epoch is not None:
        initial_epoch = args.continue_from_epoch
        print(f"Continuing from epoch {initial_epoch}")

    history = model.fit(
        X_train, y_train,
        epochs=FINETUNE_CONFIG['epochs'],
        batch_size=FINETUNE_CONFIG['batch_size'],
        validation_data=(X_valid, y_valid),
        verbose=1,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60 + "\n")

    model_path = args.model_path if args.model_path else FINETUNE_CONFIG['checkpoint_path']
    model.save(model_path)
    print(f"Fine-tuned model saved to: {model_path}")

    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60 + "\n")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\n" + "="*60)
    print("PLOTTING TRAINING HISTORY")
    print("="*60 + "\n")

    plot_path = 'results/finetuning_history.png'
    plot_training_history(history, save_path=plot_path, show=False)

    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE")
    print("="*60)
    print(f"\nModel saved to: {model_path}")
    print(f"Training history plot saved to: {plot_path}")
    print(f"Final test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nNext steps:")
    print("  1. Evaluate model: python evaluate.py")
    print("  2. Make predictions: python predict.py --image <path_to_image>")
    print("="*60 + "\n")


def main():
    """Main function to parse arguments and run fine-tuning."""
    parser = argparse.ArgumentParser(
        description='Fine-tune PATTLite model for Facial Expression Recognition'
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
        '--pretrained-model',
        type=str,
        default='checkpoints/model_initial.h5',
        help='Path to pretrained model from initial training'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help=f'Path to save fine-tuned model (default: {FINETUNE_CONFIG["checkpoint_path"]})'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of fine-tuning epochs (default: {FINETUNE_CONFIG["epochs"]})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size (default: {FINETUNE_CONFIG["batch_size"]})'
    )

    parser.add_argument(
        '--continue-from-epoch',
        type=int,
        default=None,
        help='Continue training from specific epoch'
    )

    args = parser.parse_args()

    if args.epochs is not None:
        FINETUNE_CONFIG['epochs'] = args.epochs

    if args.batch_size is not None:
        FINETUNE_CONFIG['batch_size'] = args.batch_size

    finetune(args)


if __name__ == '__main__':
    main()
