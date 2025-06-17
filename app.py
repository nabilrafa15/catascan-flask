from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
from datetime import datetime
import tensorflow as tf
from PIL import Image
import numpy as np

# Load environment variables
load_dotenv()

# Setup Flask and DB
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load Keras Model (BUKAN tf.saved_model.load)
MODEL_PATH = os.getenv("MODEL_PATH", "final_improved_cataract_model/final_improved_cataract_model")
model = tf.keras.models.load_model(MODEL_PATH)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database Result table
class Result(db.Model):
    __tablename__ = 'Results'
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String)
    prediction = db.Column(db.String)
    explanation = db.Column(db.Text)
    createdAt = db.Column(db.DateTime, default=datetime.utcnow)
    updatedAt = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    UserId = db.Column(db.Integer)

@app.route('/')
def homepage():
    return {"message": "✅ Catascan Flask aktif. Gunakan POST /predict"}

# Preprocessing sama seperti saat training
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64), resample=Image.LANCZOS)  
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "image wajib diisi"}), 400

    file = request.files['image']
    user_id = request.form.get('user_id')

    filename = file.filename
    relative_path = os.path.join(UPLOAD_FOLDER, filename).replace("\\", "/")
    file.save(relative_path)
    public_url = f"http://localhost:5000/{relative_path}"

    input_img = preprocess_image(relative_path)
    prediction_values = model.predict(input_img)
    
    label_map = ['mature', 'immature', 'normal']
    print("Raw prediction values:", prediction_values)
    pred_index = np.argmax(prediction_values)
    pred_label = label_map[pred_index]

    confidence = {
        label_map[i]: float(score) for i, score in enumerate(prediction_values[0])
    }

    explanation_dict = {
        'mature': 'Lensa mengalami kekeruhan total (mature cataract).',
        'immature': 'Kekeruhan sebagian pada lensa (immature cataract).',
        'normal': 'Tidak ditemukan indikasi katarak.'
    }
    explanation = explanation_dict.get(pred_label, 'Tidak diketahui')

    result = Result(
        image_path=relative_path,
        prediction=pred_label,
        explanation=explanation,
        UserId=user_id
    )
    db.session.add(result)
    db.session.commit()

    return jsonify({
        "prediction": pred_label,
        "explanation": explanation,
        "confidence_scores": confidence,
        "photoUrl": public_url,
        "image_path": relative_path
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
