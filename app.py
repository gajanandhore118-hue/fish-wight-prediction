from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    # Create input array
    input_data = []

    for col in columns:
        if col in data:
            input_data.append(float(data[col]))
        else:
            input_data.append(0)

    prediction = model.predict([input_data])[0]

    return render_template("index.html", prediction_text=f"Predicted Weight: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)