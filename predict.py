"""
Prediction script for PATTLite model.

This script performs inference on single images or batches.
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import DATA_CONFIG, PREDICT_CONFIG


def load_and_preprocess_image(image_path, img_size=120):
    """
    Load and preprocess a single image.

    Args:
        image_path (str): Path to the image
        img_size (int): Target image size

    Returns:
        numpy.ndarray: Preprocessed image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))

    return image


def predict_single_image(model, image_path, img_size=120, class_labels=None):
    """
    Predict emotion for a single image.

    Args:
        model: Trained Keras model
        image_path (str): Path to the image
        img_size (int): Image size for preprocessing
        class_labels (dict): Dictionary mapping class indices to labels

    Returns:
        tuple: (predicted_class, predicted_label, probabilities)
    """
    image = load_and_preprocess_image(image_path, img_size)

    image_batch = np.expand_dims(image, axis=0)

    predictions = model.predict(image_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    predicted_proba = predictions[0][predicted_class]

    if class_labels:
        predicted_label = class_labels.get(predicted_class, f"Class {predicted_class}")
    else:
        predicted_label = f"Class {predicted_class}"

    return predicted_class, predicted_label, predictions[0]


def predict_batch(model, image_paths, img_size=120, class_labels=None):
    """
    Predict emotions for a batch of images.

    Args:
        model: Trained Keras model
        image_paths (list): List of image paths
        img_size (int): Image size for preprocessing
        class_labels (dict): Dictionary mapping class indices to labels

    Returns:
        list: List of (image_path, predicted_class, predicted_label, probabilities) tuples
    """
    results = []

    for image_path in image_paths:
        try:
            pred_class, pred_label, probabilities = predict_single_image(
                model, image_path, img_size, class_labels
            )
            results.append((image_path, pred_class, pred_label, probabilities))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append((image_path, None, None, None))

    return results


def visualize_prediction(image_path, predicted_class, predicted_label,
                        probabilities, class_labels=None, save_path=None):
    """
    Visualize prediction with probabilities.

    Args:
        image_path (str): Path to the image
        predicted_class (int): Predicted class index
        predicted_label (str): Predicted class label
        probabilities (numpy.ndarray): Class probabilities
        class_labels (dict): Dictionary mapping class indices to labels
        save_path (str): Path to save the visualization
    """
    image = load_and_preprocess_image(image_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {predicted_label}\n'
                  f'Confidence: {probabilities[predicted_class]:.2%}',
                  fontsize=14, fontweight='bold')

    classes = range(len(probabilities))
    if class_labels:
        labels = [class_labels.get(i, f"Class {i}") for i in classes]
    else:
        labels = [f"Class {i}" for i in classes]

    colors = ['green' if i == predicted_class else 'steelblue' for i in classes]

    ax2.barh(labels, probabilities, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])

    for i, (label, prob) in enumerate(zip(labels, probabilities)):
        ax2.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def predict(args):
    """
    Run prediction.

    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("PATTLITE MODEL - PREDICTION")
    print("="*60 + "\n")

    model_path = args.model_path if args.model_path else PREDICT_CONFIG['model_path']

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py or finetune.py")
        return

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!\n")

    class_labels = PREDICT_CONFIG['emotion_labels']
    img_size = PREDICT_CONFIG['img_size']

    if args.image:
        print(f"Predicting for image: {args.image}")

        pred_class, pred_label, probabilities = predict_single_image(
            model, args.image, img_size, class_labels
        )

        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\nImage: {args.image}")
        print(f"Predicted Class: {pred_class}")
        print(f"Predicted Label: {pred_label}")
        print(f"Confidence: {probabilities[pred_class]:.2%}")

        print("\nClass Probabilities:")
        for class_idx, prob in enumerate(probabilities):
            label = class_labels.get(class_idx, f"Class {class_idx}")
            marker = " <--" if class_idx == pred_class else ""
            print(f"  {label}: {prob:.4f} ({prob*100:.2f}%){marker}")

        print("="*60 + "\n")

        if args.visualize:
            save_path = args.output if args.output else None
            visualize_prediction(
                args.image, pred_class, pred_label,
                probabilities, class_labels, save_path
            )

    elif args.image_folder:
        print(f"Predicting for images in folder: {args.image_folder}")

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []

        for file in os.listdir(args.image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.image_folder, file))

        if not image_paths:
            print(f"No images found in {args.image_folder}")
            return

        print(f"Found {len(image_paths)} images")

        results = predict_batch(model, image_paths, img_size, class_labels)

        print("\n" + "="*60)
        print("BATCH PREDICTION RESULTS")
        print("="*60 + "\n")

        for img_path, pred_class, pred_label, probabilities in results:
            if pred_class is not None:
                print(f"{os.path.basename(img_path)}: {pred_label} "
                      f"({probabilities[pred_class]:.2%})")
            else:
                print(f"{os.path.basename(img_path)}: Failed")

        print("\n" + "="*60 + "\n")

    else:
        print("Error: Please provide either --image or --image-folder")


def main():
    """Main function to parse arguments and run prediction."""
    parser = argparse.ArgumentParser(
        description='Predict facial expressions using PATTLite model'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help=f'Path to trained model (default: {PREDICT_CONFIG["model_path"]})'
    )

    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to a single image for prediction'
    )

    parser.add_argument(
        '--image-folder',
        type=str,
        default=None,
        help='Path to folder containing images for batch prediction'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize prediction with probabilities'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save visualization (only with --visualize)'
    )

    args = parser.parse_args()

    predict(args)


if __name__ == '__main__':
    main()
