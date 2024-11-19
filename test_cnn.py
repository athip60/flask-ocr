from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = load_model('models_cnn/cnn_model_mnist_digits.h5')

def predict_digit(image_path):
    image = Image.open(image_path).convert('L').resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # plt.imshow(image_array.reshape(28, 28), cmap='gray')
    # plt.title("Input Image")
    # plt.axis('off')
    # plt.show()
    
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

image_paths = [
    "test/digits/0-1.png",
    "test/digits/1-1.png",
    "test/digits/2-1.png",
    "test/digits/2-2.png",
    "test/digits/2-3.png",
    "test/digits/3-1.png",
    "test/digits/3-2.png",
    "test/digits/3-3.png",
    "test/digits/3-4.png",
    "test/digits/3-5.png",
    "test/digits/3-6.png",
    "test/digits/4-1.png",
    "test/digits/4-2.png",
    "test/digits/4-3.png",
    "test/digits/5-1.png",
    "test/digits/5-2.png",
    "test/digits/7-1.png",
    "test/digits/7-2.png",
    "test/digits/8-1.png",
    "test/digits/9-1.png",
    "test/digits/9-2.png"
]

for image_path in image_paths:
    digit = predict_digit(image_path)
    print(f"{image_path}: {digit}")
