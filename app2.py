import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penyakit Daun Jagung", page_icon="🌽")

# --- LOAD MODEL ---
# Pastikan file 'model_jagung_v1.h5' ada di folder yang sama
@st.cache_resource # Menggunakan cache agar model tidak di-load ulang setiap interaksi
def load_my_model():
    if os.path.exists('model_jagung_v1.h5'):
        return tf.keras.models.load_model('model_jagung_v1.h5')
    else:
        return None

model = load_my_model()
CLASSES = ['Hawar (Blight)', 'Karat (Rust)', 'Sehat (Healthy)']

# --- UI STREAMLIT ---
st.title("🌽 Deteksi Penyakit Daun Jagung")
st.write("Unggah foto daun jagung untuk mendiagnosis kondisinya secara otomatis menggunakan AI.")

# Sidebar untuk informasi tambahan
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.info("Aplikasi ini menggunakan model MobileNetV2 untuk mengklasifikasikan 3 kategori: Hawar, Karat, dan Sehat.")
    if model is None:
        st.error("⚠️ Model 'model_jagung_v1.h5' tidak ditemukan. Pastikan Anda sudah menjalankan proses training terlebih dahulu.")

# --- FITUR UPLOAD GAMBAR ---
uploaded_file = st.file_uploader("Pilih gambar daun jagung...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    if model is not None:
        if st.button('Mulai Analisis'):
            with st.spinner('Sedang menganalisis...'):
                # 1. Preprocessing Gambar
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                
                # Menangani gambar RGBA (4 channel) jika ada
                if img_array.shape[-1] == 4:
                    img_array = img_array[..., :3]
                
                img_array = np.expand_dims(img_array, axis=0)

                # 2. Prediksi
                predictions = model.predict(img_array)
                score = predictions[0]
                class_idx = np.argmax(score)
                confidence = score[class_idx] * 100

                # 3. Tampilkan Hasil
                st.divider()
                st.subheader(f"Hasil Prediksi: **{CLASSES[class_idx]}**")
                
                # Progress bar untuk tingkat keyakinan
                st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
                st.progress(int(confidence))

                # Tips berdasarkan hasil
                if CLASSES[class_idx] == 'Sehat (Healthy)':
                    st.success("Tanaman Anda terlihat sehat! Tetap jaga kelembapan dan nutrisi tanah.")
                else:
                    st.warning(f"Terdeteksi gejala {CLASSES[class_idx]}. Segera lakukan pengecekan pada area lahan dan konsultasikan dengan ahli agronomi.")
    else:
        st.error("Gagal melakukan prediksi karena model tidak tersedia.")