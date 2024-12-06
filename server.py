from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import time  # Simulating training delay

from src.pipeline import CustomData, InferencePipeline, TrainingPipeline


app = Flask(__name__)
CORS(app)

inference_pipeline = InferencePipeline()
training_pipeline = TrainingPipeline()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-api", methods=["GET", "POST"])
def predict():
    
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            request.form.get("gender"),
            request.form.get("ethnicity"),
            request.form.get("parental_level_of_education"),
            request.form.get("lunch"),
            request.form.get("test_preparation_course"),
            request.form.get("reading_score"),
            request.form.get("writing_score")
        ).get_data_as_dataframe()

        predictions = inference_pipeline.predict(data)

        return render_template("home.html", results=predictions[0])
    
@app.route("/train-model-api", methods=["GET", "POST"])
def train_model():
    try:
        training_pipeline.run_pipeline()
        time.sleep(2)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
@app.route("/train", methods=["GET"])
def render_train_page():
    return render_template("train.html")

    
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")