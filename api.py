from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- KONFIGURASI ---
MODEL_PATH = 'model_jagung_v1.h5'
CLASSES = ['Hawar (Blight)', 'Karat (Rust)', 'Sehat (Healthy)']
IMG_SIZE = (224, 224)

# --- LOAD MODEL ---
# Kita load model di luar route agar hanya ter-load sekali saat server jalan
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Model {MODEL_PATH} berhasil dimuat.")
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {str(e)}")
    model = None

def prepare_image(image, target_size):
    """Mengubah gambar menjadi format yang siap diprediksi model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah ada file yang dikirim
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang dikirim'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    try:
        # Membaca gambar dari stream (tanpa simpan ke disk)
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        
        # Preprocessing
        processed_image = prepare_image(image, IMG_SIZE)

        # Prediksi
        predictions = model.predict(processed_image)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        # Response JSON
        result = {
            'success': True,
            'prediction': CLASSES[class_idx],
            'confidence': f"{confidence * 100:.2f}%",
            'class_index': int(class_idx),
            'message': "Analisis berhasil diselesaikan"
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    # Gunakan host='0.0.0.0' agar bisa diakses oleh perangkat mobile di jaringan yang sama
    app.run(host='0.0.0.0', port=5000, debug=True)