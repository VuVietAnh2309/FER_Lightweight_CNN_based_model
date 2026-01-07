"""
PATTLite Model Architecture for Facial Expression Recognition.

This module implements the PATTLite model which combines:
- MobileNet backbone for feature extraction
- Patch extraction module using separable convolutions
- Self-attention mechanism
- Classification head

The model supports both training and fine-tuning modes.
"""

import tensorflow as tf
from tensorflow import keras
from config import DATA_CONFIG, MODEL_CONFIG


def create_data_augmentation_layer(image_size=224):
    """
    Create data augmentation layer.

    Args:
        image_size (int): Target image size after resizing

    Returns:
        tf.keras.Sequential: Data augmentation layer
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Resizing(image_size, image_size),
        tf.keras.layers.RandomFlip(mode='horizontal'),
        tf.keras.layers.RandomContrast(factor=0.3)
    ], name="augmentation")

    return data_augmentation


def create_patch_extraction_layer(filters=256):
    """
    Create patch extraction module using separable convolutions.

    Args:
        filters (int): Number of filters for convolution layers

    Returns:
        tf.keras.Sequential: Patch extraction layer
    """
    patch_extraction = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(
            filters, kernel_size=4, strides=4,
            padding='same', activation='relu'
        ),
        tf.keras.layers.SeparableConv2D(
            filters, kernel_size=2, strides=2,
            padding='valid', activation='relu'
        ),
        tf.keras.layers.Conv2D(
            filters, kernel_size=1, strides=1,
            padding='valid', activation='relu'
        )
    ], name='patch_extraction')

    return patch_extraction


def create_pre_classification_layer(units=32):
    """
    Create pre-classification dense layer with batch normalization.

    Args:
        units (int): Number of units in dense layer

    Returns:
        tf.keras.Sequential: Pre-classification layer
    """
    pre_classification = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization()
    ], name='pre_classification')

    return pre_classification


def create_base_model(input_shape=(224, 224, 3), output_layer_index=-29):
    """
    Create MobileNet base model for feature extraction.

    Args:
        input_shape (tuple): Input shape for the backbone
        output_layer_index (int): Layer index to extract features from

    Returns:
        tf.keras.Model: Base model for feature extraction
    """
    backbone = tf.keras.applications.mobilenet.MobileNet(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    backbone.trainable = False

    base_model = tf.keras.Model(
        backbone.input,
        backbone.layers[output_layer_index].output,
        name='base_model'
    )

    return base_model


def create_pattlite_model(num_classes=8, img_shape=(120, 120, 3),
                          mode='train', dropout=0.1, spatial_dropout=None):
    """
    Create the complete PATTLite model for facial expression recognition.

    Args:
        num_classes (int): Number of output classes
        img_shape (tuple): Input image shape
        mode (str): 'train' for initial training, 'finetune' for fine-tuning
        dropout (float): Dropout rate for classification head
        spatial_dropout (float): Spatial dropout rate (only for fine-tuning)

    Returns:
        tf.keras.Model: Complete PATTLite model
    """
    if num_classes is None:
        num_classes = DATA_CONFIG['num_classes']

    image_size = MODEL_CONFIG['resize_size']
    patch_filters = MODEL_CONFIG['patch_filters']
    pre_classification_units = MODEL_CONFIG['pre_classification_units']

    input_layer = tf.keras.Input(shape=img_shape, name='universal_input')

    data_augmentation = create_data_augmentation_layer(image_size)
    preprocess_input = tf.keras.applications.mobilenet.preprocess_input

    base_model = create_base_model(
        input_shape=MODEL_CONFIG['backbone_input_shape'],
        output_layer_index=MODEL_CONFIG['base_model_output_layer']
    )

    patch_extraction = create_patch_extraction_layer(filters=patch_filters)

    self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')

    pre_classification = create_pre_classification_layer(units=pre_classification_units)

    prediction_layer = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name='classification_head'
    )

    x = input_layer
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = patch_extraction(x)

    if spatial_dropout is not None and mode == 'finetune':
        x = tf.keras.layers.SpatialDropout2D(spatial_dropout)(x)

    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = pre_classification(x)
    x = self_attention([x, x])

    if mode == 'finetune':
        x = tf.keras.layers.Dropout(dropout)(x)

    outputs = prediction_layer(x)

    model_name = 'train-head' if mode == 'train' else 'finetune-backbone'
    model = tf.keras.Model(inputs=input_layer, outputs=outputs, name=model_name)

    return model


def unfreeze_base_model(model, num_layers_to_unfreeze=59):
    """
    Unfreeze layers in the base model for fine-tuning.

    Args:
        model (tf.keras.Model): The model to modify
        num_layers_to_unfreeze (int): Number of layers to unfreeze from the end

    Returns:
        tf.keras.Model: Model with unfrozen layers
    """
    base_model = model.get_layer('base_model')

    base_model.trainable = True

    fine_tune_from = len(base_model.layers) - num_layers_to_unfreeze

    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])

    print(f"\nBase model layers unfrozen for fine-tuning:")
    print(f"  Total layers: {len(base_model.layers)}")
    print(f"  Trainable layers: {trainable_count}")
    print(f"  Frozen layers: {len(base_model.layers) - trainable_count}")

    return model


def compile_model(model, learning_rate=1e-3, global_clipnorm=3.0,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy']):
    """
    Compile the model with optimizer, loss, and metrics.

    Args:
        model (tf.keras.Model): Model to compile
        learning_rate (float): Learning rate for optimizer
        global_clipnorm (float): Global gradient clipping norm
        loss (str): Loss function name
        metrics (list): List of metrics to track

    Returns:
        tf.keras.Model: Compiled model
    """
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        global_clipnorm=global_clipnorm
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


def get_model_summary(model):
    """
    Print detailed model summary.

    Args:
        model (tf.keras.Model): Model to summarize
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60 + "\n")

    model.summary()

    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    print("\n" + "="*60)
    print("MODEL PARAMETERS")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("="*60 + "\n")


if __name__ == '__main__':
    print("Testing model creation...\n")

    print("Creating model for initial training...")
    model = create_pattlite_model(
        num_classes=DATA_CONFIG['num_classes'],
        img_shape=DATA_CONFIG['img_shape'],
        mode='train',
        dropout=0.1
    )

    model = compile_model(model, learning_rate=1e-3)

    get_model_summary(model)

    print("\n" + "="*60)
    print("Creating model for fine-tuning...")
    print("="*60 + "\n")

    model_ft = create_pattlite_model(
        num_classes=DATA_CONFIG['num_classes'],
        img_shape=DATA_CONFIG['img_shape'],
        mode='finetune',
        dropout=0.2,
        spatial_dropout=0.2
    )

    model_ft = unfreeze_base_model(model_ft, num_layers_to_unfreeze=59)
    model_ft = compile_model(model_ft, learning_rate=1e-5)

    get_model_summary(model_ft)

    print("Model creation test completed successfully!")
