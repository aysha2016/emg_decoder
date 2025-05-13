import time
import numpy as np
import sounddevice as sd
import pyttsx3
import threading
import queue
from collections import deque
from .preprocess import EMGPreprocessor
from .model import load_saved_model
import tensorflow as tf

class EMGStreamer:
    def __init__(self, model_path='models/emg_decoder.h5', 
                 fs=1000, n_channels=8, window_size=200,
                 device_id=None, tts_enabled=True):
        """
        Initialize EMG streamer for real-time processing.
        
        Args:
            model_path: Path to the trained model
            fs: Sampling frequency
            n_channels: Number of EMG channels
            window_size: Window size for processing
            device_id: Audio device ID for EMG acquisition
            tts_enabled: Whether to enable text-to-speech feedback
        """
        self.fs = fs
        self.n_channels = n_channels
        self.window_size = window_size
        self.device_id = device_id
        self.is_streaming = False
        
        # Initialize components
        self.preprocessor = EMGPreprocessor(
            fs=fs, 
            window_size=window_size,
            n_channels=n_channels
        )
        
        # Load model
        try:
            self.model = load_saved_model(model_path)
        except FileNotFoundError:
            print("No model found, using dummy model for testing")
            self.model = None
        
        # Initialize TTS engine
        self.tts_enabled = tts_enabled
        if tts_enabled:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_queue = queue.Queue()
            self.tts_thread = threading.Thread(target=self._tts_worker)
            self.tts_thread.daemon = True
            self.tts_thread.start()
        
        # Prediction buffer for smoothing
        self.pred_buffer = deque(maxlen=5)
        self.class_names = ["Rest", "Flex", "Extend", "Grip", "Release"]
        
        # Callback queue for processed data
        self.callback_queue = queue.Queue()
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Process incoming data
        if self.is_streaming:
            # Convert to numpy array and reshape
            data = indata.reshape(-1, self.n_channels)
            
            # Add samples to preprocessor
            for sample in data:
                self.preprocessor.add_sample(sample)
    
    def _tts_worker(self):
        """Background thread for TTS processing"""
        while True:
            try:
                text = self.tts_queue.get()
                if text is None:
                    break
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except queue.Empty:
                continue
    
    def _process_predictions(self, predictions):
        """Process model predictions with smoothing"""
        # Add prediction to buffer
        self.pred_buffer.append(predictions)
        
        # Average predictions
        avg_pred = np.mean(self.pred_buffer, axis=0)
        predicted_class = np.argmax(avg_pred)
        confidence = avg_pred[predicted_class]
        
        # Only provide feedback if confidence is high enough
        if confidence > 0.7:
            class_name = self.class_names[predicted_class]
            if self.tts_enabled and class_name != "Rest":
                self.tts_queue.put(f"Detected {class_name}")
            return class_name, confidence
        return "Rest", confidence
    
    def start_streaming(self):
        """Start streaming and processing EMG data"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.preprocessor.start_processing()
        
        # Start audio stream
        self.stream = sd.InputStream(
            channels=self.n_channels,
            samplerate=self.fs,
            callback=self._audio_callback,
            device=self.device_id
        )
        self.stream.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_streaming(self):
        """Stop streaming and processing"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        self.preprocessor.stop_processing()
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if self.tts_enabled:
            self.tts_queue.put(None)  # Signal TTS thread to stop
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_streaming:
            try:
                # Get processed window from preprocessor
                processed, features = self.preprocessor.feature_queue.get_nowait()
                
                if self.model is not None:
                    # Make prediction
                    predictions = self.model.predict(
                        np.expand_dims(processed, axis=0),
                        verbose=0
                    )[0]
                    
                    # Process predictions
                    class_name, confidence = self._process_predictions(predictions)
                    
                    # Put results in callback queue
                    self.callback_queue.put({
                        'class': class_name,
                        'confidence': confidence,
                        'features': features,
                        'timestamp': time.time()
                    })
            except queue.Empty:
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
                continue
    
    def get_latest_prediction(self):
        """Get the latest prediction from the callback queue"""
        try:
            return self.callback_queue.get_nowait()
        except queue.Empty:
            return None

def stream_emg_simulator(data, callback, interval=0.01):
    """
    Simulate EMG streaming for testing purposes.
    
    Args:
        data: Array of EMG samples
        callback: Function to call with each sample
        interval: Time between samples in seconds
    """
    for sample in data:
        callback(sample)
        time.sleep(interval)

def test_streamer():
    """Test the EMG streamer with simulated data"""
    # Generate some dummy EMG data
    n_samples = 1000
    n_channels = 8
    dummy_data = np.random.randn(n_samples, n_channels)
    
    # Create streamer instance
    streamer = EMGStreamer(tts_enabled=False)
    
    # Define callback
    def process_sample(sample):
        streamer.preprocessor.add_sample(sample)
    
    # Start streaming
    streamer.start_streaming()
    
    # Simulate data streaming
    stream_emg_simulator(dummy_data, process_sample)
    
    # Stop streaming
    streamer.stop_streaming()

if __name__ == "__main__":
    test_streamer()
