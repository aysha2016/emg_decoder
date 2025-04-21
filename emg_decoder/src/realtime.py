
import time

class_names = ["Rest", "Flex", "Extend"]

def stream_emg_simulator(data, callback, interval=0.01):
    for sample in data:
        callback(sample)
        time.sleep(interval)
