from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# โหลดโมเดล
model = load_model('mnist_cnn_model.h5')

# สร้าง Flask App
app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    # รับไฟล์ภาพจากผู้ใช้งาน
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('L').resize((28, 28))  # แปลงเป็น Grayscale และปรับขนาด
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape ให้เข้ากับโมเดล

    predicted_probabilities = model.predict(image_array)
    predicted_class = int(np.argmax(predicted_probabilities))

    return jsonify({
        "predicted_class": predicted_class,
        "probabilities": predicted_probabilities.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
