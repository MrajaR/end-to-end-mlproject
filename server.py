from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd

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
        return render_template("predict.html")
    else:
        data = CustomData(
            str(request.form.get("gender")).lower(),
            str(request.form.get("ethnicity")),
            str(request.form.get("parental_level_of_education")).lower(),
            str(request.form.get("lunch")).lower(),
            str(request.form.get("test_preparation_course")).lower(),
            float(request.form.get("reading_score")),
            float(request.form.get("writing_score"))
        ).get_data_as_dataframe()

        predictions = inference_pipeline.predict(data)

        return render_template("predict.html", results=predictions[0])
    
@app.route("/train-model-api", methods=["GET", "POST"])
def train_model():
    try:
        training_pipeline.run_pipeline()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
@app.route("/train", methods=["GET"])
def render_train_page():
    return render_template("train.html")

    
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")