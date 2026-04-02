import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penyakit Daun Jagung", page_icon="🌽")

# --- LOAD MODEL ---
@st.cache_resource 
def load_my_model():
    if os.path.exists('model_jagung_v1.h5'):
        return tf.keras.models.load_model('model_jagung_v1.h5')
    else:
        return None

model = load_my_model()
# Label harus sesuai dengan urutan index saat training
CLASSES = ['Hawar (Blight)', 'Karat (Rust)', 'Sehat (Healthy)']

# --- UI STREAMLIT ---
st.title("🌽 Deteksi Penyakit Daun Jagung")
st.write("Unggah foto daun jagung untuk mendiagnosis kondisinya secara otomatis.")

with st.sidebar:
    st.header("Tentang Aplikasi")
    st.info("Aplikasi ini menggunakan MobileNetV2. Akurasi sangat bergantung pada kualitas pencahayaan dan kejelasan gambar daun.")
    if model is None:
        st.error("⚠️ Model 'model_jagung_v1.h5' tidak ditemukan.")

# --- FITUR UPLOAD GAMBAR ---
uploaded_file = st.file_uploader("Pilih gambar daun jagung...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    if model is not None:
        if st.button('Mulai Analisis'):
            with st.spinner('Sedang menganalisis...'):
                # 1. Preprocessing
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                
                if img_array.shape[-1] == 4:
                    img_array = img_array[..., :3]
                
                img_array = np.expand_dims(img_array, axis=0)

                # 2. Prediksi
                predictions = model.predict(img_array)
                score = predictions[0]
                class_idx = np.argmax(score)
                confidence = score[class_idx] * 100

                st.divider()

                # --- VALIDASI APAKAH INI DAUN JAGUNG ---
                # Jika skor tertinggi di bawah 65%, kemungkinan besar bukan daun jagung
                THRESHOLD = 65.0 

                if confidence < THRESHOLD:
                    st.error(f"⚠️ **Gambar Tidak Dikenali.**")
                    st.warning(f"Tingkat keyakinan hanya {confidence:.2f}%. Mohon unggah foto daun jagung yang lebih jelas dan dekat.")
                else:
                    # 3. Tampilkan Hasil Jika Valid
                    st.subheader(f"Hasil Prediksi: **{CLASSES[class_idx]}**")
                    st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
                    st.progress(int(confidence))

                    # Perbaikan logika pengecekan string (disesuaikan dengan isi CLASSES)
                    if 'Sehat' in CLASSES[class_idx]:
                        st.success("Tanaman Anda terlihat sehat! Tetap jaga kelembapan dan nutrisi tanah.")
                    else:
                        st.warning(f"Terdeteksi gejala {CLASSES[class_idx]}. Segera lakukan pengecekan lahan.")
    else:
        st.error("Gagal melakukan prediksi karena model tidak tersedia.")
