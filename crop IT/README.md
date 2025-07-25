## Smart Farming Prediction System
A machine learning-powered web application built using Flask that classifies and predicts farming-related outcomes based on input features. The backend leverages scikit-learn models such as Decision Tree, Random Forest, and SVM to deliver predictions and insights for better farming decisions.

## Features
Upload CSV data for training.

Data preprocessing and scaling.

Train multiple ML models:

Decision Tree

Random Forest

Support Vector Machine (SVM)

View accuracy, classification reports, and confusion matrices.

Visualization using Matplotlib & Seaborn.

Backend implemented in Flask.

## Project Structure
bash
Copy
Edit
.
├── app.py                 # Flask backend (if present separately)
├── new.ipynb             # Main notebook with code logic
├── templates/            # HTML templates (if Flask is integrated)
│   └── index.html        # Main UI
├── static/               # Static files like CSS/JS
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation (this file)
## Machine Learning Models
Your project uses:

 Decision Tree

 Random Forest

 Support Vector Machine (SVM)

Each model is trained and evaluated using scikit-learn, with metrics like:

Accuracy Score

Confusion Matrix

Classification Report

## Requirements
You can install the required packages using:

bash
Copy
Edit
pip install -r requirements.txt
Dependencies include:

Flask

Pandas

NumPy

scikit-learn

matplotlib

seaborn

## How to Run
Clone the repository.

Make sure the dataset (smart farming1.csv) is placed at the correct path.

Run the notebook (new.ipynb) or convert it to a Flask app (app.py) if needed.

If Flask is integrated, start the server:

bash
Copy
Edit
python app.py
Then go to http://127.0.0.1:5000/ in your browser.

## Dataset
The project uses a CSV file:

bash
Copy
Edit
D:/farming/smart farming1.csv
Make sure to place it in the same or updated path used by your code.

## Visualizations
The project generates:

Heatmaps of correlation

Confusion matrices

Model performance metrics

## Functions Included
read_input(): Reads CSV data.

print_data_insights(): Prints summary stats and head.

scale_data(): Applies standard scaling.

train_decision_tree(), train_random_forest(), train_svm(): Train and evaluate classifiers.

Visualization utilities.