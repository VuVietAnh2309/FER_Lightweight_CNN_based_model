# PATTLite: Lightweight CNN-based Model for Facial Expression Recognition

A lightweight deep learning model for facial expression recognition using MobileNet backbone with patch extraction and self-attention mechanisms, trained on the RAF-DB dataset.

## Model Architecture

The PATTLite model consists of:

- **Data Augmentation Layer**: Random horizontal flip and contrast adjustment
- **Backbone**: MobileNet (pretrained on ImageNet) for feature extraction
- **Patch Extraction Module**: Separable convolutions for efficient feature processing
- **Self-Attention Mechanism**: For capturing important spatial relationships
- **Classification Head**: Dense layers with batch normalization

### Training Strategy

The model uses a two-phase training approach:

1. **Initial Training**: Train with frozen backbone to learn task-specific features
2. **Fine-tuning**: Unfreeze backbone layers for end-to-end optimization

## Project Structure

```
Code/
├── config.py                 # Configuration and hyperparameters
├── dataset.py                # Data loading and preprocessing
├── model.py                  # Model architecture
├── train.py                  # Initial training script
├── finetune.py              # Fine-tuning script
├── evaluate.py              # Evaluation script
├── predict.py               # Inference script
├── utils/
│   ├── __init__.py
│   ├── callbacks.py         # Training callbacks
│   ├── visualization.py     # Plotting utilities
│   └── metrics.py           # Evaluation metrics
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Code
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

This project uses the RAF-DB (Real-world Affective Faces Database) dataset.

1. Download the RAF-DB dataset
2. Organize the dataset in the following structure:
```
dataset/
├── train/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   ├── 6/
│   └── 7/
└── test/
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    ├── 6/
    └── 7/
```

3. Update the dataset path in [config.py](config.py):
```python
DATA_CONFIG = {
    'dataset_path': '/path/to/your/dataset',
    'dataset_name': 'raf-db-dataset/DATASET',
    ...
}
```

## Usage

### 1. Configuration

Edit [config.py](config.py) to adjust hyperparameters:

- Dataset paths and image settings
- Model architecture parameters
- Training hyperparameters (learning rate, batch size, etc.)
- Fine-tuning settings
- Evaluation and prediction configurations

### 2. Initial Training

Train the model with frozen backbone:

```bash
python train.py
```

Optional arguments:
```bash
python train.py --dataset-path /path/to/dataset \
                --dataset-name raf-db-dataset/DATASET \
                --model-path checkpoints/my_model.h5 \
                --epochs 100 \
                --batch-size 8
```

### 3. Fine-tuning

Fine-tune the model by unfreezing backbone layers:

```bash
python finetune.py --pretrained-model checkpoints/model_initial.h5
```

Optional arguments:
```bash
python finetune.py --pretrained-model checkpoints/model_initial.h5 \
                   --model-path checkpoints/model_final.h5 \
                   --epochs 65 \
                   --batch-size 8
```

### 4. Evaluation

Evaluate the trained model:

```bash
python evaluate.py --model-path checkpoints/model_finetune.h5
```

Optional arguments:
```bash
python evaluate.py --model-path checkpoints/model_finetune.h5 \
                   --split test \
                   --save-predictions
```

Evaluation outputs:
- Overall metrics (accuracy, precision, recall, F1-score)
- Per-class accuracy
- Confusion matrix
- Classification report
- Sample predictions visualization

### 5. Prediction

Predict on a single image:

```bash
python predict.py --image /path/to/image.jpg --visualize
```

Predict on a folder of images:

```bash
python predict.py --image-folder /path/to/images/
```

Optional arguments:
```bash
python predict.py --image /path/to/image.jpg \
                  --model-path checkpoints/model_finetune.h5 \
                  --visualize \
                  --output results/prediction.png
```

## Configuration

All hyperparameters are centralized in [config.py](config.py):

### Data Configuration
- Image size: 120x120
- Number of classes: 8
- Validation split: 10%

### Training Configuration (Initial Training)
- Epochs: 100
- Batch size: 8
- Learning rate: 1e-3
- Dropout: 0.1
- Early stopping patience: 5

### Fine-tuning Configuration
- Epochs: 65
- Batch size: 8
- Learning rate: 1e-5
- Dropout: 0.2
- Spatial dropout: 0.2
- Layers to unfreeze: 59
- Early stopping patience: 20

## Results

The model achieves the following performance on the RAF-DB dataset:

- **Test Accuracy**: ~94% (after fine-tuning)
- **Training Time**: ~2-3 hours (depends on hardware)

Results are saved in the `results/` directory:
- Training history plots
- Confusion matrices
- Classification reports
- Sample predictions

## Model Checkpoints

Models are saved in the `checkpoints/` directory:

- `model_initial.h5`: Model after initial training
- `model_finetune.h5`: Model after fine-tuning

## Logs

Training logs are saved in `logs/fit/` for TensorBoard visualization:

```bash
tensorboard --logdir logs/fit
```

## Testing Individual Modules

Each module can be tested independently:

```bash
# Test configuration
python config.py

# Test dataset loading
python dataset.py

# Test model creation
python model.py

# Test callbacks
python utils/callbacks.py
```

## Tips for Best Results

1. **Data Augmentation**: Adjust augmentation parameters in [model.py](model.py) for your dataset
2. **Learning Rate**: Use learning rate finder to determine optimal learning rate
3. **Batch Size**: Increase batch size if you have more GPU memory
4. **Early Stopping**: Adjust patience based on training stability
5. **Class Weights**: Enable class weights for imbalanced datasets

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch size in [config.py](config.py)
- Reduce image size (not recommended as it affects accuracy)

### Low Accuracy
- Train for more epochs
- Adjust learning rate
- Enable class weights for imbalanced datasets
- Check data preprocessing pipeline

### Slow Training
- Use GPU instead of CPU
- Increase batch size
- Use mixed precision training

## Requirements

- TensorFlow 2.12.0
- OpenCV 4.8.0
- NumPy 1.23.5
- scikit-learn 1.3.0
- Matplotlib 3.7.2
- seaborn 0.12.2

## License

This project is for research and educational purposes.

## Citation

If you use this code in your research, please cite:

```
@article{pattlite2024,
  title={PATTLite: Lightweight CNN-based Model for Facial Expression Recognition},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2024}
}
```

## Acknowledgments

- RAF-DB dataset creators
- TensorFlow and Keras teams
- MobileNet architecture authors

## Contact

For questions or issues, please open an issue in the repository or contact [your-email@example.com].
