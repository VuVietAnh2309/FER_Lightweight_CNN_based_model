"""
Visualization utilities for training and evaluation.

This module provides functions to plot training history, confusion matrices,
and sample predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(history, save_path=None, show=True):
    """
    Plot training and validation accuracy and loss.

    Args:
        history: Keras History object or dict with 'accuracy', 'val_accuracy', 'loss', 'val_loss'
        save_path (str): Path to save the plot. If None, won't save
        show (bool): Whether to display the plot
    """
    if hasattr(history, 'history'):
        history = history.history

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['accuracy']) + 1)

    axes[0].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[1].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None,
                         normalize=False, show=True):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the plot
        normalize (bool): Whether to normalize the confusion matrix
        show (bool): Whether to display the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)),
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_sample_predictions(images, true_labels, pred_labels, class_names=None,
                           num_samples=16, save_path=None, show=True):
    """
    Plot sample images with true and predicted labels.

    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names (list): List of class names
        num_samples (int): Number of samples to plot
        save_path (str): Path to save the plot
        show (bool): Whether to display the plot
    """
    num_samples = min(num_samples, len(images))

    indices = np.random.choice(len(images), num_samples, replace=False)

    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i, idx in enumerate(indices):
        img = images[idx]

        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        true_label = true_labels[idx]
        pred_label = pred_labels[idx]

        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
        else:
            true_name = str(true_label)
            pred_name = str(pred_label)

        axes[i].imshow(img)
        axes[i].axis('off')

        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_name}\nPred: {pred_name}',
                         fontsize=10, color=color, fontweight='bold')

    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(y, class_names=None, title='Class Distribution',
                           save_path=None, show=True):
    """
    Plot class distribution as a bar chart.

    Args:
        y: Labels array
        class_names (list): List of class names
        title (str): Plot title
        save_path (str): Path to save the plot
        show (bool): Whether to display the plot
    """
    unique, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(10, 6))

    bars = plt.bar(unique, counts, color='steelblue', alpha=0.7, edgecolor='black')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    if class_names:
        plt.xticks(unique, [class_names[i] for i in unique])

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_training_history()")
    print("  - plot_confusion_matrix()")
    print("  - plot_sample_predictions()")
    print("  - plot_class_distribution()")
