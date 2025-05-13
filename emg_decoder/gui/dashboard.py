import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import time
import os
from src.realtime import EMGStreamer
from src.model import load_saved_model, convert_to_tflite
import json

# Page config
st.set_page_config(
    page_title="MIND: Neural EMG Decoder",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'streamer' not in st.session_state:
    st.session_state.streamer = None
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'history' not in st.session_state:
    st.session_state.history = {
        'timestamp': [],
        'class': [],
        'confidence': [],
        'features': []
    }

# Sidebar controls
st.sidebar.title("🧠 MIND Controls")

# Model selection
model_path = st.sidebar.selectbox(
    "Select Model",
    ["models/emg_decoder.h5", "models/cnn_lstm_model.h5"],
    index=0
)

# Device selection
try:
    import sounddevice as sd
    devices = sd.query_devices()
    input_devices = [d['name'] for d in devices if d['max_input_channels'] > 0]
    device_id = st.sidebar.selectbox("Select Input Device", input_devices)
except:
    st.sidebar.warning("No audio devices found")
    device_id = None

# Streaming controls
col1, col2 = st.sidebar.columns(2)
start_button = col1.button("▶️ Start", disabled=st.session_state.is_streaming)
stop_button = col2.button("⏹️ Stop", disabled=not st.session_state.is_streaming)

# TTS toggle
tts_enabled = st.sidebar.checkbox("Enable TTS Feedback", value=True)

# Export options
st.sidebar.title("Export Options")
export_tflite = st.sidebar.button("Export to TFLite")
if export_tflite:
    try:
        model = load_saved_model(model_path)
        convert_to_tflite(model)
        st.sidebar.success("Model exported to TFLite format!")
    except Exception as e:
        st.sidebar.error(f"Export failed: {str(e)}")

# Main content
st.title("🧠 MIND: Neural EMG Decoder Dashboard")

# Create columns for metrics
col1, col2, col3, col4 = st.columns(4)

# Initialize placeholders
metrics_placeholder = st.empty()
plot_placeholder = st.empty()
history_placeholder = st.empty()

def update_metrics(prediction):
    """Update the metrics display"""
    if prediction is None:
        return
    
    with metrics_placeholder.container():
        st.markdown("### Current Prediction")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Class", prediction['class'])
        with col2:
            st.metric("Confidence", f"{prediction['confidence']:.2%}")
        with col3:
            st.metric("Latency", f"{(time.time() - prediction['timestamp'])*1000:.1f}ms")
        with col4:
            st.metric("Buffer Size", len(st.session_state.history['timestamp']))

def update_plots(prediction):
    """Update the real-time plots"""
    if prediction is None:
        return
    
    # Update history
    st.session_state.history['timestamp'].append(prediction['timestamp'])
    st.session_state.history['class'].append(prediction['class'])
    st.session_state.history['confidence'].append(prediction['confidence'])
    st.session_state.history['features'].append(prediction['features'])
    
    # Keep only last 100 points
    max_points = 100
    for key in st.session_state.history:
        st.session_state.history[key] = st.session_state.history[key][-max_points:]
    
    # Create plots
    with plot_placeholder.container():
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("EMG Signal", "Prediction Confidence"),
            vertical_spacing=0.1,
            heights=[0.7, 0.3]
        )
        
        # Add EMG signal plot
        if len(st.session_state.history['features']) > 0:
            features = np.array(st.session_state.history['features'])
            for i in range(min(8, features.shape[1])):
                fig.add_trace(
                    go.Scatter(
                        y=features[:, i],
                        name=f"Channel {i+1}",
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # Add confidence plot
        if len(st.session_state.history['confidence']) > 0:
            fig.add_trace(
                go.Scatter(
                    y=st.session_state.history['confidence'],
                    name="Confidence",
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

def update_history():
    """Update the prediction history table"""
    if len(st.session_state.history['timestamp']) == 0:
        return
    
    with history_placeholder.container():
        st.markdown("### Prediction History")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Time': [datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3] 
                    for ts in st.session_state.history['timestamp']],
            'Class': st.session_state.history['class'],
            'Confidence': [f"{conf:.2%}" for conf in st.session_state.history['confidence']]
        })
        
        # Display table
        st.dataframe(df, use_container_width=True)

# Handle streaming controls
if start_button and not st.session_state.is_streaming:
    try:
        st.session_state.streamer = EMGStreamer(
            model_path=model_path,
            device_id=device_id,
            tts_enabled=tts_enabled
        )
        st.session_state.streamer.start_streaming()
        st.session_state.is_streaming = True
        st.sidebar.success("Streaming started!")
    except Exception as e:
        st.sidebar.error(f"Failed to start streaming: {str(e)}")

if stop_button and st.session_state.is_streaming:
    if st.session_state.streamer:
        st.session_state.streamer.stop_streaming()
    st.session_state.is_streaming = False
    st.sidebar.info("Streaming stopped")

# Main loop
if st.session_state.is_streaming and st.session_state.streamer:
    while True:
        prediction = st.session_state.streamer.get_latest_prediction()
        if prediction:
            update_metrics(prediction)
            update_plots(prediction)
            update_history()
        time.sleep(0.01)  # Small sleep to prevent CPU spinning

# Display instructions when not streaming
if not st.session_state.is_streaming:
    st.info("""
    ### Getting Started
    1. Select your input device from the sidebar
    2. Choose a model to use
    3. Click 'Start' to begin streaming
    4. Monitor the real-time predictions and signal
    5. Use 'Stop' to end the session
    
    The dashboard will show:
    - Real-time EMG signal visualization
    - Current prediction and confidence
    - Historical predictions
    - System metrics
    """)
