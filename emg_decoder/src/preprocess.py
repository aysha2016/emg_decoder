
import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def preprocess_emg(data, fs=1000, lowcut=20.0, highcut=450.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered = lfilter(b, a, data, axis=0)
    normalized = (filtered - np.mean(filtered, axis=0)) / np.std(filtered, axis=0)
    return normalized
