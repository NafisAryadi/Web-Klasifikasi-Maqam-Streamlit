import librosa
import numpy as np, math
from filterSuara import butter_bandpass_filter

lowcut= 300
highcut= 3400

# Fungsi normalisasi per file (zero mean, unit variance)
def normalize_sample(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + 1e-6)

# --- Create fixed-size windows --- (Update untuk Audio yang Di-upload)
def create_windows_and_fix_shape(x, time_steps=500, stride=100):
    x = normalize_sample(x)
    if x.shape[1] < time_steps:
        repeats = math.ceil(time_steps / x.shape[1])
        x = np.tile(x, (1, repeats))
        x = x[:, :time_steps]
    windows = []
    for i in range(0, x.shape[1] - time_steps + 1, stride):
        window = x[:, i:i + time_steps]
        if window.shape[1] == time_steps:
            window = np.expand_dims(window, axis=-1)
            windows.append(window)
    return windows

def process_chroma_with_windowing(file_path, time_steps=500, stride=100):
    # Memuat file audio
    audio_data, sr = librosa.load(file_path, sr=22050)  # Atur sr sesuai dengan pelatihan model Anda    
    
    # Filter audio (sesuai kebutuhan)
    filtered_audio = butter_bandpass_filter(audio_data, lowcut , highcut, sr)
    
    # Ekstraksi fitur (misalnya, MFCC atau Chroma) - Anda dapat menyesuaikan fitur yang ingin digunakan
    chromas = librosa.feature.chroma_stft(y=filtered_audio, sr=sr)
    
    # Membagi audio menjadi jendela dengan ukuran time_steps dan stride yang ditentukan
    windows = create_windows_and_fix_shape(chromas, time_steps, stride)
    return np.array(windows)

def predict_audio_with_chroma(file_path, model, time_steps=1000, stride=100):
    windows = process_chroma_with_windowing(file_path, time_steps, stride)
    
    # Sesuaikan bentuk fitur untuk prediksi model
    #windows = np.expand_dims(windows, axis=0)  # Menambahkan dimensi batch
    prediction = model.predict(windows)
    avg_probability = prediction.mean(axis=0)
    top3_idx = np.argsort(avg_probability)[-3:][::-1]
    return [(int(idx), float(avg_probability[idx])) for idx in top3_idx]

