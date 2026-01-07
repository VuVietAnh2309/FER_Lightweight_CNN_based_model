"""
Evaluation script for PATTLite model.

This script evaluates a trained model and generates comprehensive metrics.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from config import DATA_CONFIG, EVAL_CONFIG, create_directories
from dataset import prepare_data
from utils.metrics import print_evaluation_metrics, save_metrics_to_file
from utils.visualization import plot_confusion_matrix, plot_sample_predictions


def evaluate(args):
    """
    Evaluate trained model.

    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("PATTLITE MODEL - EVALUATION")
    print("="*60 + "\n")

    create_directories()

    model_path = args.model_path if args.model_path else EVAL_CONFIG['model_path']

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py or finetune.py")
        return

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!\n")

    print("Loading test data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, _ = prepare_data(
        dataset_path=args.dataset_path if args.dataset_path else DATA_CONFIG['dataset_path'],
        dataset_name=args.dataset_name if args.dataset_name else DATA_CONFIG['dataset_name'],
        validation_split=DATA_CONFIG['validation_split'],
        random_state=DATA_CONFIG['random_state'],
        img_size=DATA_CONFIG['img_size'],
        compute_class_weights=False
    )

    if args.split == 'test':
        X_eval, y_eval = X_test, y_test
        split_name = "Test"
    elif args.split == 'validation':
        X_eval, y_eval = X_valid, y_valid
        split_name = "Validation"
    elif args.split == 'train':
        X_eval, y_eval = X_train, y_train
        split_name = "Train"
    else:
        print(f"Invalid split: {args.split}. Using test set.")
        X_eval, y_eval = X_test, y_test
        split_name = "Test"

    print(f"\nEvaluating on {split_name} set ({len(X_eval)} samples)...")

    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60 + "\n")

    test_loss, test_acc = model.evaluate(
        X_eval, y_eval,
        batch_size=EVAL_CONFIG['batch_size'],
        verbose=1
    )

    print(f"\n{split_name} Loss: {test_loss:.4f}")
    print(f"{split_name} Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60 + "\n")

    y_pred_proba = model.predict(X_eval, batch_size=EVAL_CONFIG['batch_size'], verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(f"Predictions generated for {len(y_pred)} samples")

    class_names = [f"Class {i}" for i in range(DATA_CONFIG['num_classes'])]

    print_evaluation_metrics(y_eval, y_pred, class_names=class_names)

    if EVAL_CONFIG['save_results']:
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60 + "\n")

        results_dir = EVAL_CONFIG['results_dir']
        os.makedirs(results_dir, exist_ok=True)

        report_path = os.path.join(results_dir, f'classification_report_{args.split}.txt')
        save_metrics_to_file(y_eval, y_pred, report_path, class_names=class_names)

        cm_path = os.path.join(results_dir, f'confusion_matrix_{args.split}.png')
        print(f"\nGenerating confusion matrix...")
        plot_confusion_matrix(
            y_eval, y_pred,
            class_names=class_names,
            save_path=cm_path,
            normalize=False,
            show=False
        )

        cm_norm_path = os.path.join(results_dir, f'confusion_matrix_{args.split}_normalized.png')
        plot_confusion_matrix(
            y_eval, y_pred,
            class_names=class_names,
            save_path=cm_norm_path,
            normalize=True,
            show=False
        )

        if args.save_predictions:
            pred_path = os.path.join(results_dir, f'sample_predictions_{args.split}.png')
            print(f"Generating sample predictions visualization...")
            plot_sample_predictions(
                X_eval, y_eval, y_pred,
                class_names=class_names,
                num_samples=16,
                save_path=pred_path,
                show=False
            )

        predictions_file = os.path.join(results_dir, f'predictions_{args.split}.npz')
        np.savez(
            predictions_file,
            y_true=y_eval,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba
        )
        print(f"Predictions saved to: {predictions_file}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"Dataset split: {split_name}")
    print(f"Samples evaluated: {len(y_eval)}")
    print(f"Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Loss: {test_loss:.4f}")

    if EVAL_CONFIG['save_results']:
        print(f"\nResults saved to: {results_dir}")

    print("="*60 + "\n")


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate PATTLite model for Facial Expression Recognition'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help=f'Path to trained model (default: {EVAL_CONFIG["model_path"]})'
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
        '--split',
        type=str,
        default='test',
        choices=['train', 'validation', 'test'],
        help='Dataset split to evaluate on (default: test)'
    )

    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save sample predictions visualization'
    )

    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
