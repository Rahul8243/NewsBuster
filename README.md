📰 Fake News Detector (Machine Learning + Flask)

A machine-learning powered Fake News Detection system that classifies news as REAL or FAKE using TF-IDF vectorization and Logistic Regression.
Includes a simple and clean Flask web app for real-time predictions.

⭐ Summary

A Flask-based Fake News Detector trained on TF-IDF features that predicts whether a news article is real or fake with probability scores.

📌 Description

This project provides a full end-to-end Fake News Detection pipeline:

Data loading, preprocessing & text cleaning

TF-IDF vectorizer (1–2 n-grams)

Logistic Regression model training

Evaluation with accuracy, F1-score & confusion matrix

Saving model + vectorizer using joblib

Flask-based frontend where users can paste any news text to get predictions

It is designed to behave realistically in real-world conditions and generalize well on unseen news.

📂 Project Structure
Fake-News-Detector/
│
├── app.py                     # Flask web application for prediction
├── train.py                   # ML pipeline: training + evaluation
│
├── models/
│   ├── model.pkl              # Saved Logistic Regression model
│   ├── vectorizer.pkl         # Saved TF-IDF vectorizer
│   └── confusion_matrix.png   # Evaluation visualization
│
├── templates/
│   └── index.html             # Front-end UI for prediction
│
├── data/
│   └── merged_news.csv        # Dataset used for training
│
└── README.md                  # Documentation

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model
python train.py


This will generate:

models/model.pkl

models/vectorizer.pkl

models/confusion_matrix.png

4️⃣ Run the Flask app
python app.py


Navigate to:
➡️ http://127.0.0.1:5000

🧠 Model Details

Algorithm: Logistic Regression

Feature Extraction: TF-IDF Vectorizer (unigram + bigram)

Train/Test Split: 80/20 (stratified)

Metrics: Accuracy, Precision, Recall, F1-score

Output:

Label: REAL / FAKE

Probability of FAKE news

🖥️ Web App Features

Clean and simple user interface

Enter any news headline or paragraph

Get prediction instantly

Probability score included

📊 Evaluation

The training script automatically generates a confusion matrix showing:

True Real

True Fake

Misclassifications

Saved at:

models/confusion_matrix.png

🔮 Future Improvements

Add XGBoost / SVM for better accuracy

Develop API endpoints

Use transformer models (BERT)

Build a Streamlit dashboard

Rahul Raj
📧 Email: rahulrajmahi611@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/rahul-kumar-ab8843198/

🐙 GitHub: https://github.com/Rahul8243
