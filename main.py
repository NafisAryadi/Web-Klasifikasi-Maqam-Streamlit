import os
import sys
import csv
import time
from datetime import datetime
import streamlit as st
import soundfile as sf
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))
from preprocessing import predict_audio_with_chroma


HISTORY_CSV = "history.csv"
if not os.path.exists(HISTORY_CSV):
    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "timestamp", "predicted_class", "confidence"])

def gen_timestamp_filename(suffix=".wav"):
    ts = int(time.time())
    return f"{ts}{suffix}"

def save_buffer_as_wav(buffer, sr=16000):
    fn = gen_timestamp_filename(".wav")
    os.makedirs("recordings", exist_ok=True)
    path = os.path.join("recordings", fn)
    sf.write(path, buffer, sr)
    return path

def save_uploaded_file(upl):
    fn = gen_timestamp_filename(os.path.splitext(upl.name)[1])
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", fn)
    with open(path, "wb") as f: f.write(upl.getbuffer())
    return path

model_chroma = tf.keras.models.load_model("model/chroma_model.h5")

st.header(":blue[Klasifikasi] Maqam Bacaan :green[Al-'Quran]")

lowcut= 300
highcut= 3400


source = st.radio("Sumber audio:", ["Upload", "Rekam"])

file_path = None
if source == "Upload":
    upl = st.file_uploader("Pilih file untuk diunggah", type="wav")
    if upl is not None:
        file_path = save_uploaded_file(upl)
        st.success(f"Disimpan sebagai: `{file_path}`")
        st.audio(file_path)  # playback

else:  # Rekam
    st.write("🎤 Klik tombol ► untuk mulai merekam, klik ■ apabila selesai.")
    audio_bytes = st.audio_input("Rekam suara Anda di sini")  # ← widget baru
    if audio_bytes:
        fn = f"{int(time.time())}.wav"
        os.makedirs("recordings", exist_ok=True)
        file_path = os.path.join("recordings", fn)
        # audio_bytes adalah io.BytesIO
        data = audio_bytes.read() if hasattr(audio_bytes, "read") else audio_bytes
        with open(file_path, "wb") as f:
            f.write(data)
        st.success(f"Rekaman berhasil diunggah")
        st.audio(file_path)

if file_path is not None:
    if st.button('Prediksi'):
            with st.spinner('Proses prediksi sedang berlangsung, harap tunggu...'):
                time.sleep(3)
                predicted_class, max_probability = predict_audio_with_chroma(file_path, model_chroma)
                class_names = ['Ajam', 'Bayat', 'Hijaz', 'Kurd', 'Nahawand', 'Rast', 'Saba', 'Seka']
                st.write(f"Hasil Klasifikasi: {class_names[predicted_class]}")
                st.write(f"Kemungkinan Prediksi: {max_probability:.2f}")

                ori_name = upl.name if source == "Upload" else "Rekaman"

                with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        ori_name,
                        datetime.now().isoformat(sep=" ", timespec="seconds"),
                        class_names[predicted_class],
                        f"{max_probability:.4f}"
                    ])