**🔍 Overview**

NewsBuster is an AI-powered fake news detection web application built with Flask and Machine Learning (NLP).
It analyzes input text from news articles and classifies them as REAL or FAKE based on linguistic and statistical features.

This project demonstrates how AI can help in detecting misinformation using modern Natural Language Processing and Machine Learning techniques.

**✨ Features**

🧠 AI-based Fake News Classifier — Uses a trained Logistic Regression model.

🧹 Text Cleaning Pipeline — Removes URLs, punctuation, and noise before prediction.

💻 Interactive Web Interface — Built with Flask and modern HTML/CSS design.

📊 Confidence Score Bar — Displays model’s probability for better interpretability.

📱 Responsive UI — Works smoothly on both desktop and mobile.

🧩 Explainable ML Stack — Easy to understand, modify, and retrain.

**🧰 Tech Stack**
Layer	Tools Used
Backend	Python, Flask
Machine Learning	Scikit-learn, Pandas, NumPy
Vectorization	TF-IDF
Frontend	HTML5, CSS3 (Dark UI)
Model Type	Logistic Regression
Serialization	Joblib

**🧪 How It Works**

The user enters or pastes a piece of news text.

The app cleans and preprocesses the text (removing URLs, special chars, etc.).

Text is converted into a TF-IDF vector.

The trained ML model predicts whether it’s “REAL” or “FAKE.”

The result and confidence probability are displayed in the web UI.

**⚙️ Installation & Setup**

1️⃣ Clone this repository
git clone https://github.com/Rahul8243/NewsBuster.git
cd NewsBuster

2️⃣ Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model (if not already present)

Make sure you have your dataset ready, then run:
python train.py


This will generate:

models/model.pkl
models/vectorizer.pkl

5️⃣ Run the Flask app
python app.py


**Then open your browser and go to 👉 http://localhost:5000**

**🧩 Project Structure**
NewsBuster/
│
├── app.py                 # Flask app (main backend)
├── train.py               # Script to train model
├── requirements.txt       # Python dependencies
├── models/
│   ├── model.pkl          # Trained ML model
│   └── vectorizer.pkl     # TF-IDF vectorizer
├── templates/
│   └── index.html         # Frontend UI
├── static/                # (Optional) CSS, JS, or image files
└── README.md              # Documentation

**🧠 Example Output**
Input	Prediction	Confidence
“Government announces new vaccine drive tomorrow.”	REAL	0.94
“Aliens landed in Paris last night, officials confirm.”	FAKE	0.87

**👨‍💻 Developer**
Rahul Kumar
📧 rahulrajmahi611@gmail.com

**🌐 GitHub Profile**
https://github.com/Rahul8243

**🌟 Acknowledgements**
https://scikit-learn.org/stable/
https://flask.palletsprojects.com/en/stable/
https://www.kaggle.com/
 