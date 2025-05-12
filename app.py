from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image
import joblib
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model1 = load_model('kapeng_model.keras')
model2 = load_model('kapeng_model.keras')
model3 = load_model('kapeng_model.keras')
models = {'model1': model1, 'model2': model2, 'model3': model3}

morph_model = joblib.load('morph_model.joblib')
morph_encoder = joblib.load('label_encoder.joblib')
class_names = ['Murking', 'Saludo', 'Katys', 'Tunying']

def prepare_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def extract_morph_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area if rect_area else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area else 0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter else 0
    return [area, perimeter, aspect_ratio, extent, solidity, circularity]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    model_choice = request.form['model']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    if model_choice == 'morph':
        feats = extract_morph_features(file_path)
        if feats is None:
            return jsonify({'prediction': 'Feature extraction failed'})
        pred = morph_model.predict([feats])
        predicted_class = morph_encoder.inverse_transform(pred)[0]
    elif model_choice in models:
        img_tensor = prepare_image(file_path)
        predictions = models[model_choice].predict(img_tensor)
        predicted_class = class_names[np.argmax(predictions)]
    else:
        predicted_class = 'Invalid model selected'

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
