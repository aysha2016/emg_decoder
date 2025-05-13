# 🧠 MIND: Neural EMG Decoder

A real-time neural interface that decodes motor signals from EMG data using deep learning. Built with Keras and TensorFlow, featuring a Streamlit dashboard for real-time monitoring and TensorFlow Lite export for microcontroller deployment.

## 🌟 Features

- **Real-time EMG Signal Processing**
  - Multi-channel EMG data acquisition
  - Bandpass filtering and signal normalization
  - Time and frequency domain feature extraction
  - Real-time prediction with confidence smoothing

- **Deep Learning Architecture**
  - CNN-LSTM hybrid model for temporal-spatial feature extraction
  - Parallel CNN branches for multi-scale feature learning
  - Bidirectional LSTM layers for improved temporal modeling
  - Dropout and batch normalization for robust training

- **Interactive Dashboard**
  - Real-time EMG signal visualization
  - Live prediction monitoring with confidence scores
  - Historical data tracking and analysis
  - Device selection and model management
  - Text-to-Speech feedback for motor intent

- **Microcontroller Deployment**
  - TensorFlow Lite model export
  - Model quantization and optimization
  - Platform-specific optimizations (Arduino, ESP32)
  - Arduino header file generation
  - Metadata export for model information

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.12.0+
- Keras 2.12.0+
- Streamlit 1.22.0+
- Other dependencies listed in `requirements.txt`

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/aysha2016/emg_decoder.git
cd emg_decoder
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

## 💻 Usage

### Running the Dashboard

Start the Streamlit dashboard for real-time monitoring:
```bash
python main.py dashboard
```

The dashboard provides:
- Real-time EMG signal visualization
- Current prediction and confidence metrics
- Historical prediction tracking
- Device selection and model management
- Export options for microcontroller deployment

### Model Export

Export your trained model to TensorFlow Lite format:
```bash
# Basic export
python main.py export --model models/emg_decoder.h5

# Export with specific options
python main.py export --model models/emg_decoder.h5 \
                     --platform arduino \
                     --generate-header \
                     --no-quantize
```

Export options:
- `--model`: Path to input Keras model
- `--out-dir`: Output directory for exported files
- `--platform`: Target platform (arduino, esp32, generic)
- `--no-quantize`: Disable model quantization
- `--no-optimize`: Disable model optimization
- `--generate-header`: Generate Arduino header file

## 🏗️ Project Structure

```
emg_decoder/
├── src/
│   ├── model.py          # CNN-LSTM model architecture
│   ├── preprocess.py     # EMG signal preprocessing
│   └── realtime.py       # Real-time processing
├── gui/
│   └── dashboard.py      # Streamlit dashboard
├── microcontroller/
│   └── export_model.py   # TFLite export utilities
├── models/               # Saved models
├── main.py              # CLI entry point
└── requirements.txt     # Project dependencies
```

## 🔧 Model Architecture

The EMG decoder uses a hybrid CNN-LSTM architecture:
- **Input**: EMG signals (200 time steps × 8 channels)
- **CNN Layers**: 
  - Parallel branches with different kernel sizes
  - Batch normalization and max pooling
  - Dropout for regularization
- **LSTM Layers**:
  - Bidirectional LSTM for temporal modeling
  - Multiple layers for hierarchical feature learning
- **Output**: Softmax classification (5 classes)

## 📊 Signal Processing

The system implements:
- Bandpass filtering (20-450 Hz)
- Signal normalization
- Time domain features:
  - Mean, standard deviation
  - Maximum absolute value
  - Integrated EMG
- Frequency domain features:
  - Power in delta, theta, alpha, beta, and gamma bands

 
