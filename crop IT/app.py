# pip install flask numpy pandas joblib matplotlib scikit-learn

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the model (ensure your model supports predict_proba)
model = joblib.load("models/SVM.pkl")

# Replace this with your actual crop class labels (in correct order)
crop_list = model.classes_.tolist()  # Automatically gets crop labels from the model

def plot_crop_probabilities(probabilities, crop_names):
    best_index = np.argmax(probabilities)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(crop_names, probabilities, color='skyblue')
    bars[best_index].set_color('green')  # Highlight the best crop

    ax.set_ylabel("Suitability Score")
    ax.set_xlabel("Crops")
    ax.set_title("Crop Suitability Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded = base64.b64encode(img.read()).decode('utf-8')
    plt.close()
    return encoded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(key)) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(final_features)[0]
            chart = plot_crop_probabilities(proba, crop_list)
        else:
            chart = None

        return render_template('index.html', prediction=prediction, chart=chart)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
