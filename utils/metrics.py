"""
Evaluation metrics for model performance.

This module provides functions to compute and display various
evaluation metrics for classification tasks.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy score.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        float: Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true, y_pred, average='weighted'):
    """
    Compute precision score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average (str): Averaging method ('weighted', 'macro', 'micro')

    Returns:
        float: Precision score
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true, y_pred, average='weighted'):
    """
    Compute recall score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average (str): Averaging method ('weighted', 'macro', 'micro')

    Returns:
        float: Recall score
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1_score(y_true, y_pred, average='weighted'):
    """
    Compute F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average (str): Averaging method ('weighted', 'macro', 'micro')

    Returns:
        float: F1 score
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_per_class_accuracy(y_true, y_pred, num_classes=None):
    """
    Compute per-class accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes (int): Number of classes. If None, infer from data

    Returns:
        dict: Dictionary mapping class index to accuracy
    """
    if num_classes is None:
        num_classes = max(max(y_true), max(y_pred)) + 1

    per_class_acc = {}

    for class_idx in range(num_classes):
        mask = y_true == class_idx
        if mask.sum() > 0:
            class_correct = (y_pred[mask] == class_idx).sum()
            class_total = mask.sum()
            per_class_acc[class_idx] = class_correct / class_total
        else:
            per_class_acc[class_idx] = 0.0

    return per_class_acc


def get_classification_report(y_true, y_pred, class_names=None, output_dict=False):
    """
    Generate classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): List of class names
        output_dict (bool): If True, return as dict instead of string

    Returns:
        str or dict: Classification report
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )


def print_evaluation_metrics(y_true, y_pred, class_names=None):
    """
    Print comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): List of class names
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    accuracy = compute_accuracy(y_true, y_pred)
    precision = compute_precision(y_true, y_pred, average='weighted')
    recall = compute_recall(y_true, y_pred, average='weighted')
    f1 = compute_f1_score(y_true, y_pred, average='weighted')

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    per_class_acc = compute_per_class_accuracy(y_true, y_pred)

    print(f"\nPer-Class Accuracy:")
    for class_idx, acc in per_class_acc.items():
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)

    report = get_classification_report(y_true, y_pred, class_names=class_names)
    print(report)

    print("="*60 + "\n")


def save_metrics_to_file(y_true, y_pred, filepath, class_names=None):
    """
    Save evaluation metrics to a text file.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        filepath (str): Path to save the metrics
        class_names (list): List of class names
    """
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")

        accuracy = compute_accuracy(y_true, y_pred)
        precision = compute_precision(y_true, y_pred, average='weighted')
        recall = compute_recall(y_true, y_pred, average='weighted')
        f1 = compute_f1_score(y_true, y_pred, average='weighted')

        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n\n")

        per_class_acc = compute_per_class_accuracy(y_true, y_pred)

        f.write("Per-Class Accuracy:\n")
        for class_idx, acc in per_class_acc.items():
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
            f.write(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)\n")

        f.write("\n" + "="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")

        report = get_classification_report(y_true, y_pred, class_names=class_names)
        f.write(report)

        f.write("\n" + "="*60 + "\n")

    print(f"Metrics saved to: {filepath}")


if __name__ == '__main__':
    print("Metrics utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - compute_accuracy()")
    print("  - compute_precision()")
    print("  - compute_recall()")
    print("  - compute_f1_score()")
    print("  - compute_per_class_accuracy()")
    print("  - get_classification_report()")
    print("  - print_evaluation_metrics()")
    print("  - save_metrics_to_file()")
