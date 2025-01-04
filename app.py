from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import face_recognition
import numpy as np
import joblib
from captcha.image import ImageCaptcha
import random
import string
import os
import pandas as pd

app = Flask(__name__)

# === Yuzni Tanish (Face ID) Qismi === #

# Yuzni tanish uchun bazaviy rasm yuklash
KNOWN_IMAGE_PATH = "known_face.jpg"  # Bu sizning tanib olinadigan yuzingiz
if not os.path.exists(KNOWN_IMAGE_PATH):
    raise FileNotFoundError(f"{KNOWN_IMAGE_PATH} fayli mavjud emas!")
known_image = face_recognition.load_image_file(KNOWN_IMAGE_PATH)
known_face_encoding = face_recognition.face_encodings(known_image)[0]

def compare_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for face_encoding in face_encodings:
        match = face_recognition.compare_faces([known_face_encoding], face_encoding, tolerance=0.6)
        if match[0]:
            return True
    return False

# === Bitcoin Narxi Bashorati Qismi === #

# Modelni yuklash
MODEL_PATH = "bitcoin_model5.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} fayli mavjud emas!")
model = joblib.load(MODEL_PATH)

def generate_captcha():
    image_captcha = ImageCaptcha(width=280, height=90)
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    image = image_captcha.generate_image(captcha_text)
    image_path = f"static/{captcha_text}.png"
    image.save(image_path)
    return captcha_text, image_path

def verify_captcha(user_input, actual_captcha):
    return user_input.strip().upper() == actual_captcha

# === Flask Routing === #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face-id', methods=['GET', 'POST'])
def face_id():
    if request.method == 'POST':
        file = request.files.get('video_frame')
        if file:
            frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            verified = compare_faces(frame)
            if verified:
                return jsonify({'status': 'success', 'message': 'Yuz tanildi!'})
            return jsonify({'status': 'error', 'message': 'Yuz topilmadi yoki mos kelmadi.'})
    return render_template('face_id.html', known_image_path=KNOWN_IMAGE_PATH)

@app.route('/bitcoin-prediction', methods=['GET', 'POST'])
def bitcoin_prediction():
    if "captcha_text" not in request.cookies:
        captcha_text, captcha_image = generate_captcha()
    else:
        captcha_text = request.cookies.get('captcha_text')
        captcha_image = f"static/{captcha_text}.png"

    if request.method == 'POST':
        open_price = float(request.form.get('open_price', 0.0))
        high_price = float(request.form.get('high_price', 0.0))
        low_price = float(request.form.get('low_price', 0.0))
        volume = float(request.form.get('volume', 0.0))
        user_captcha = request.form.get('captcha', '')

        if not verify_captcha(user_captcha, captcha_text):
            return render_template('bitcoin.html', error="CAPTCHA noto‘g‘ri!", captcha_image=captcha_image)

        input_data = pd.DataFrame({
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price],
            'Volume': [volume]
        })
        prediction = model.predict(input_data)[0]
        return render_template('bitcoin.html', prediction=round(prediction, 2), captcha_image=captcha_image)

    return render_template('bitcoin.html', captcha_image=captcha_image)

if __name__ == "__main__":
    app.run(debug=True)
