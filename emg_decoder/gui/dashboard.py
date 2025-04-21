
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from src.preprocess import preprocess_emg
from src.realtime import stream_emg_simulator, class_names
from keras.utils import to_categorical

model = load_model("models/cnn_lstm_model.h5")

st.title("🧠 MIND: Neural EMG Decoder Dashboard")
st.sidebar.title("Live Controls")

start_button = st.sidebar.button("Start Streaming")
placeholder = st.empty()
plot_area = st.empty()

buffer = []
predicted = st.empty()

if start_button:
    st.sidebar.success("Streaming Started")
    
    def stream_callback(sample):
        buffer.append(sample)
        if len(buffer) >= 200:
            segment = preprocess_emg(np.array(buffer[-200:]), fs=1000)
            segment = np.expand_dims(segment, axis=0)
            prediction = model.predict(segment)
            predicted_class = np.argmax(prediction)
            label = class_names[predicted_class]
            
            predicted.markdown(f"### 🤖 Predicted: `{label}`")
            plot_area.line_chart(np.array(buffer[-200:]))

    emg_data = np.load("data/sample_emg.npy")
    stream_emg_simulator(emg_data, stream_callback)
