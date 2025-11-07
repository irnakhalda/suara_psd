import streamlit as st
import numpy as np
import joblib
import tempfile
import pandas as pd
from utils.feature_extraction import extract_features
from st_audiorec import st_audiorec

# ===== CONFIG =====
st.set_page_config(page_title="Voice Identification", page_icon="🎧", layout="centered")
st.title("🎙️ Voice Identification System")
st.write("""
Sistem ini melakukan identifikasi suara **buka/tutup** dari dua pengguna yang diizinkan:  
**user1** dan **user2**.  
Silakan rekam atau unggah file suara .wav.
""")

# ===== LOAD MODEL =====
try:
    model_user = joblib.load("models/user_model.pkl")
    model_status = joblib.load("models/status_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ===== PILIH INPUT =====
input_option = st.radio("Pilih metode input:", ["🎤 Rekam suara", "📁 Upload file .wav"])

# ===== PROSES AUDIO =====
def process_audio(audio_path):
    import librosa
    import speech_recognition as sr

    features = extract_features(audio_path)
    if not features:
        st.error("Gagal mengekstraksi fitur dari suara.")
        return

    # ===== DETEKSI SILENCE =====
    y, sr_audio = librosa.load(audio_path, sr=None)
    rms = np.mean(librosa.feature.rms(y=y))
    duration = len(y)/sr_audio
    if duration < 0.1 or rms < 1e-4:
        st.error("Tidak ada suara yang terdeteksi. Silakan rekam ulang.")
        return

    feature_df = pd.DataFrame([features])
    # Pastikan urutan kolom sama seperti training
    X = feature_df[feature_cols].to_numpy().reshape(1, -1)
    st.write(f"Jumlah fitur yang dikirim ke model: {X.shape[1]}")

    # ===== PREDIKSI USER & STATUS =====
    user_pred_proba = model_user.predict_proba(X)
    status_pred_proba = model_status.predict_proba(X)
    user_pred = np.argmax(user_pred_proba)
    status_pred = np.argmax(status_pred_proba)
    user_confidence = np.max(user_pred_proba)
    status_confidence = np.max(status_pred_proba)

    # Mapping ke label
    user_label = f"user{user_pred + 1}"
    status_label = "buka" if status_pred == 0 else "tutup"

    # ===== SPEECH TO TEXT =====
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="id-ID")
            st.subheader("📝 Teks yang terdeteksi")
            st.write(text)

            # Koreksi prediksi status berdasarkan kata
            text_lower = text.lower()
            if "buka" in text_lower:
                status_label = "buka"
            elif "tutup" in text_lower:
                status_label = "tutup"
    except Exception as e:
        st.warning(f"Teks tidak dapat dikenali: {e}")

    # ===== Hasil =====
    st.subheader("📊 Hasil Prediksi")
    st.success(f"**Prediksi:** {user_label} — {status_label}")
    st.info(f"User: {user_label} | Status: {status_label}")

    if status_label == "buka":
        st.write(f"🔊 Suara mirip **{user_label} saat membuka** sesuatu.")
    else:
        st.write(f"🔊 Suara mirip **{user_label} saat menutup** sesuatu.")

    # ===== Lihat fitur =====
    with st.expander("📈 Lihat fitur yang diekstraksi"):
        st.dataframe(feature_df.T, use_container_width=True)

# ===== INPUT REKAM =====
if input_option == "🎤 Rekam suara":
    st.write("Klik tombol di bawah untuk merekam suara Anda:")
    audio_bytes = st_audiorec()
    if audio_bytes is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(audio_bytes)
            audio_path = tmpfile.name
        st.audio(audio_path, format="audio/wav")
        process_audio(audio_path)

# ===== INPUT UPLOAD =====
elif input_option == "📁 Upload file .wav":
    uploaded_file = st.file_uploader("Upload file suara (.wav):", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(uploaded_file.read())
            audio_path = tmpfile.name
        process_audio(audio_path)
