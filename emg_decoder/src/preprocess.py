import numpy as np
from scipy.signal import butter, lfilter, filtfilt, welch
from scipy.fft import fft, fftfreq
import tensorflow as tf
from collections import deque
import threading
import queue

class EMGPreprocessor:
    def __init__(self, fs=1000, lowcut=20.0, highcut=450.0, 
                 window_size=200, overlap=0.5, n_channels=8):
        """
        Initialize EMG preprocessor for real-time signal processing.
        
        Args:
            fs: Sampling frequency
            lowcut: Lower cutoff frequency for bandpass filter
            highcut: Upper cutoff frequency for bandpass filter
            window_size: Number of samples per window
            overlap: Window overlap ratio (0-1)
            n_channels: Number of EMG channels
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.window_size = window_size
        self.overlap = overlap
        self.n_channels = n_channels
        
        # Initialize bandpass filter
        self.b, self.a = self._butter_bandpass()
        
        # Buffer for real-time processing
        self.buffer = deque(maxlen=window_size)
        self.buffer_lock = threading.Lock()
        
        # Feature extraction parameters
        self.feature_queue = queue.Queue()
        self.is_processing = False
        
    def _butter_bandpass(self, order=4):
        """Design bandpass filter"""
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        return butter(order, [low, high], btype='band')
    
    def _apply_filter(self, data):
        """Apply bandpass filter to data"""
        return filtfilt(self.b, self.a, data, axis=0)
    
    def _extract_features(self, window):
        """Extract time and frequency domain features"""
        features = []
        
        # Time domain features
        features.extend([
            np.mean(window, axis=0),  # Mean
            np.std(window, axis=0),   # Standard deviation
            np.max(np.abs(window), axis=0),  # Maximum absolute value
            np.sum(np.abs(window), axis=0),  # Integrated EMG
        ])
        
        # Frequency domain features
        for ch in range(self.n_channels):
            fft_vals = np.abs(fft(window[:, ch]))
            freqs = fftfreq(len(window), 1/self.fs)
            
            # Power in different frequency bands
            delta_mask = (freqs >= 0) & (freqs < 4)
            theta_mask = (freqs >= 4) & (freqs < 8)
            alpha_mask = (freqs >= 8) & (freqs < 13)
            beta_mask = (freqs >= 13) & (freqs < 30)
            gamma_mask = (freqs >= 30) & (freqs < 100)
            
            features.extend([
                np.sum(fft_vals[delta_mask]),
                np.sum(fft_vals[theta_mask]),
                np.sum(fft_vals[alpha_mask]),
                np.sum(fft_vals[beta_mask]),
                np.sum(fft_vals[gamma_mask])
            ])
        
        return np.array(features)
    
    def preprocess_window(self, window):
        """Process a single window of EMG data"""
        # Apply bandpass filter
        filtered = self._apply_filter(window)
        
        # Normalize
        normalized = (filtered - np.mean(filtered, axis=0)) / (np.std(filtered, axis=0) + 1e-6)
        
        # Extract features
        features = self._extract_features(normalized)
        
        return normalized, features
    
    def add_sample(self, sample):
        """Add a new sample to the buffer and process if window is full"""
        with self.buffer_lock:
            self.buffer.append(sample)
            
            if len(self.buffer) == self.window_size:
                window = np.array(self.buffer)
                processed, features = self.preprocess_window(window)
                self.feature_queue.put((processed, features))
                
                # Remove overlap samples
                for _ in range(int(self.window_size * (1 - self.overlap))):
                    self.buffer.popleft()
    
    def start_processing(self):
        """Start the real-time processing thread"""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the real-time processing thread"""
        self.is_processing = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
    
    def _processing_loop(self):
        """Background processing loop"""
        while self.is_processing:
            try:
                # Process any available windows
                while not self.feature_queue.empty():
                    processed, features = self.feature_queue.get_nowait()
                    # Here you would typically send the processed data to the model
                    # or store it for later use
            except queue.Empty:
                continue
    
    @staticmethod
    def preprocess_batch(data, fs=1000, lowcut=20.0, highcut=450.0):
        """Process a batch of EMG data (for offline processing)"""
        preprocessor = EMGPreprocessor(fs, lowcut, highcut)
        processed_data = []
        features_list = []
        
        # Process data in windows
        for i in range(0, len(data) - preprocessor.window_size + 1, 
                      int(preprocessor.window_size * (1 - preprocessor.overlap))):
            window = data[i:i + preprocessor.window_size]
            processed, features = preprocessor.preprocess_window(window)
            processed_data.append(processed)
            features_list.append(features)
        
        return np.array(processed_data), np.array(features_list)

def create_tf_dataset(processed_data, labels, batch_size=32):
    """Create a TensorFlow dataset for training"""
    dataset = tf.data.Dataset.from_tensor_slices((processed_data, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
