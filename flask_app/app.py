from flask import Flask, request, render_template
from model import load_model, ModelMeta, ModelPrediction
import joblib
import os
import numpy as np
import torch

app = Flask(__name__)
model_path = os.path.join(".", "wights_raw.pickle")
scaler_path = os.path.join(".", "scaler.pkl")
print(f"Loading model from: {model_path}")
print(f"Loading scaler from: {scaler_path}")
model, meta_data = load_model(model_path)
scaler = joblib.load(scaler_path)
class_mapping = {0: "Normal", 1: "Pre-Failure", 2: "Failure"}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None
    if request.method == 'POST':
        input_data = request.form['input_data'].split(',')
        try:
            input_data = [float(x) for x in input_data]
            prediction = make_prediction(input_data)
            if isinstance(prediction, str):
                error_message = prediction
        except ValueError:
            error_message = "Please enter 16 comma-separated numerical values."
    return render_template('index.html', prediction=prediction, error_message=error_message)

def make_prediction(data):
    try:
        data_t = np.array(data, dtype=np.float32)
        if data_t.shape[0] != 16:
            return "Please enter 16 comma-separated values."

        data_t = scaler.transform(data_t.reshape(-1, 16))
        data_t = torch.tensor(data_t, dtype=torch.float32)
        preds = model(data_t).detach().cpu().numpy()[0]
        epred = np.exp(preds)
        probs = epred / epred.sum()
        pclass = np.argmax(probs)
        conf = np.max(probs)
        lbl = class_mapping[int(pclass)]
        prediction = ModelPrediction(
            predicted_class=lbl,
            predicted_label=int(pclass),
            raw_out=preds.tolist(),
            probabilities=probs.tolist(),
            confidence=conf
        )
        return prediction
    except ValueError:
        return "Invalid input. Please enter 16 comma-separated values."

@app.route('/Model', methods=['GET'])
def get_model():
    return meta_data.dict()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)