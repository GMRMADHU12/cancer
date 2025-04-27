from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = 'model/skin_disease_model.h5'
model = load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = ["actinic keratosis", "basal cell carcinoma", "melanoma", "pigmented benign keratosis"]

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Make prediction
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = round(100 * np.max(prediction), 2)

            return render_template("index.html", prediction=predicted_class, confidence=confidence, image_path=filepath)

    return render_template("index.html", prediction=None, confidence=None, image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
