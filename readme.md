# Road Network Detection from Satellite Images

This project implements a deep learning solution for detecting road networks from satellite imagery using a UNet++ architecture with EfficientNet-B4 encoder.

## Features

- UNet++ architecture with EfficientNet-B4 encoder for accurate road segmentation
- Support for both Jupyter Notebook and modular Python code approaches
- Advanced evaluation metrics including Dice coefficient and BCE loss
- Real-time training monitoring with progress bars
- Early stopping and model checkpointing
- Configurable data augmentation pipeline
- Interactive visualization tools for predictions

## Project Structure

```
roadnet_minimal/
├── data/
│   └── archive/
│       ├── train/
│       ├── valid/
│       └── test/
├── src/
│   ├── model.py          # Model architecture definition
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Evaluation metrics
│   ├── data_loader.py    # Data loading utilities
│   ├── utils.py          # Helper functions
│   └── debug.py          # Debugging utilities
├── main.py               # Entry point for modular approach
├── main.ipynb           # Jupyter notebook implementation
├── config.yaml          # Configuration file
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- segmentation-models-pytorch
- OpenCV
- NumPy
- Matplotlib
- tqdm
- PyYAML

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/roadnet_minimal.git
cd roadnet_minimal
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using Jupyter Notebook (main.ipynb)

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `main.ipynb` and run the cells sequentially.

The notebook provides an interactive environment with:
- Detailed explanations of each step
- Visualization of training data and predictions
- Real-time model performance monitoring
- Interactive parameter tuning

### Option 2: Using Modular Python Code

1. Configure your training parameters in `config.yaml`

2. Run training:
```bash
python main.py --mode train
```

3. Run evaluation:
```bash
python main.py --mode evaluate
```

## Model Training

The project supports two approaches to model training, the notebook approach is the sandbox and the main.py approach is the deployed version:
### 1. Notebook Approach (main.ipynb)
- Interactive and educational
- Step-by-step visualization
- Easy experimentation with parameters
- Great for development and debugging

### 2. Modular Approach (main.py)
- Production-ready implementation
- Configurable via YAML
- Better code organization
- Supports distributed training
- Automated logging and checkpointing

## Latest Updates

### Enhanced Evaluation Metrics
The latest version includes additional evaluation metrics:
- Dice Coefficient
- Binary Cross-Entropy Loss
- Precision and Recall
- IoU (Intersection over Union)
- Real-time visualization of predictions

### Training Improvements
- Early stopping with configurable patience
- Learning rate scheduling
- Model checkpointing
- Progress bars with real-time metrics
- Improved data augmentation pipeline

## Configuration

Key parameters in `config.yaml`:

```yaml
model:
  architecture: unetplusplus
  encoder_name: efficientnet-b4
  encoder_weights: imagenet
  classes: 1

train:
  epochs: 50
  learning_rate: 1e-4
  weight_decay: 1e-5
  early_stopping_patience: 10
```

## Model Architecture

The project uses UNet++ with EfficientNet-B4 encoder:
- Encoder: EfficientNet-B4 (pretrained on ImageNet)
- Decoder: UNet++ with skip connections
- Output: Single-channel binary segmentation mask

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.



