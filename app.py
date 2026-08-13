from flask import Flask, render_template, request
import numpy as np
import pickle

# Load trained model
with open("MLR.pkl", "rb") as f:
    m = pickle.load(f)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Read values from HTML
        rd = float(request.form["R&D"])
        admin = float(request.form["Administration"])
        marketing = float(request.form["Marketing Spend"])
        state = request.form["State"]

        # Convert state name into number
        state_mapping = {
            "New York": 0,
            "California": 1,
            "Florida": 2
        }

        state = state_mapping[state]

        # Model input
        data = np.array([[rd, admin, marketing, state]])

        # Prediction
        prediction = m.predict(data)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Profit: ${prediction:,.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)