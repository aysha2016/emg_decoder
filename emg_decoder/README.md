
# 🧠 MIND: Neural Decoder for EMG

A real-time neural interface that decodes motor signals from EMG using AI.
Built with Keras, Streamlit GUI, and microcontroller-ready exports.

## Features
- Real-time EMG prediction dashboard
- CNN-LSTM trained decoder
- Integration with TTS for speech feedback
- Export to microcontroller (TFLite)

## To Run
```bash
pip install -r requirements.txt
python main.py
```

## For Microcontroller Export
```bash
python microcontroller/export_model.py
```
