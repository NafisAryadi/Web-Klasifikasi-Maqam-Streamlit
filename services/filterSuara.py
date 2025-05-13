import os
import soundfile as sf
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(data, lowcut , highcut, fs, order=5):
    nyquist = 0.5 * fs
    low_normal_cutoff = lowcut / nyquist
    high_normal_cutoff = highcut / nyquist
    b, a = butter(order, [low_normal_cutoff, high_normal_cutoff], btype='bandpass', analog=False)
    y = filtfilt(b, a, data)
    return y

def filter_audio(audio_data, sr = 22050, lowcut=300, highcut=3400):
    filtered_audio = butter_bandpass_filter(audio_data, lowcut, highcut, sr)
    return filtered_audio
