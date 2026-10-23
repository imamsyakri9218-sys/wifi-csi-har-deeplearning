import numpy as np
from scipy.signal import butter, lfilter

def butter_lowpass_filter(data, cutoff=20.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data, axis=-1)
    return y

def preprocess_csi_signal(raw_csi_data):
    data = np.array(raw_csi_data, dtype=np.float32)
    filtered_data = butter_lowpass_filter(data)
    mean = np.mean(filtered_data, axis=(0, 2), keepdims=True)
    std = np.std(filtered_data, axis=(0, 2), keepdims=True)
    normalized_data = (filtered_data - mean) / (std + 1e-8)
    return normalized_data