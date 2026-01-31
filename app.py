from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# -------------------------
# Load trained model
# -------------------------
model = joblib.load("model.pkl")   # works if saved using joblib

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Invalid or missing JSON data"}), 400

        input_df = pd.DataFrame([{
            "age": float(data["age"]),
            "bmi": float(data["bmi"]),
            "children": int(data["children"]),
            "sex": str(data["sex"]).lower(),
            "smoker": str(data["smoker"]).lower(),
            "region": str(data["region"]).lower()
        }])

        prediction = model.predict(input_df)[0]

        return jsonify({
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
