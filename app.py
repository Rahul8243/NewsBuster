from flask import Flask, render_template, request
import joblib
import os
import re

app = Flask(__name__)

MODEL_PATH = "models/model.pkl"
VECT_PATH = "models/vectorizer.pkl"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+","", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise FileNotFoundError("Model or vectorizer not found. Run train.py first to create 'models/model.pkl'.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None
    text = ""
    if request.method == "POST":
        text = request.form.get("news_text", "")
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        pred_proba = model.predict_proba(vec)[0][1]  # probability of FAKE (1)
        result = "FAKE" if pred == 1 else "REAL"
        prob = f"{pred_proba:.3f}"
    return render_template("index.html", result=result, prob=prob, text=text)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
