from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Dictionary mapping indeks kelas ke nama kelas
dic = {
    0: 'adenocarcinoma', 
    1: 'benign',
    2: 'squamous_cell_carcinoma',
}

# Muat model yang ingin disediakan (hanya ResNet dan Inception)
model_resnet = load_model('CheckpointHisto.keras')
model_inception = load_model('CheckpointHisto.keras')

# Aktifkan predict_function
model_resnet.make_predict_function()
model_inception.make_predict_function()

# Fungsi untuk memprediksi label gambar
def predict_label(img_path, model, target_size):
    # Sesuaikan target_size dengan kebutuhan masing-masing model
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # (Jika perlu, lakukan normalisasi sesuai preprocessing yang digunakan saat training)
    p = model.predict(img)
    
    # Tentukan threshold confidence score
    threshold = 0.5  # Sesuaikan jika perlu

    confidence = np.max(p)
    if confidence < threshold:
        return "Kelas tidak ditemukan", confidence

    predicted_class = np.argmax(p, axis=1)[0]
    return dic[predicted_class], confidence

# Route untuk halaman utama
@app.route("/", methods=['GET'])
def main():
    return render_template("classification.html")

# Route untuk memproses gambar yang diupload dan pilihan model
@app.route("/submit", methods=['POST'])
def get_output():
    img = request.files['my_image']
    model_choice = request.form.get('model_choice')

    img_path = "static/" + img.filename 
    img.save(img_path)

    # Pilih model dan target size yang sesuai (hanya ResNet dan Inception)
    if model_choice == "ResNet50":
        selected_model = model_resnet
        target_size = (416, 416)
    elif model_choice == "InceptionV3":
        selected_model = model_inception
        target_size = (416, 416)
    else:
        # Jika tidak ada pilihan yang valid
        return render_template("classification.html", prediction="Model tidak ditemukan", confidence=0, img_path=img_path)

    # Lakukan prediksi dengan model yang dipilih
    prediction, confidence = predict_label(img_path, selected_model, target_size)

    return render_template("classification.html", prediction=prediction, confidence=confidence, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
