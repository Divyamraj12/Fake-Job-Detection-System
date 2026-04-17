from flask import Flask, request, render_template
import pickle
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)

# 🔥 Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔥 Prediction function
def predict_job(text):
    text_vector = vectorizer.transform([text])
    
    # Keyword feature
    keywords = ['work from home', 'earn money', 'no experience', 'urgent', 'quick money']
    count = sum([1 for word in keywords if word in text.lower()])
    
    keyword_array = np.array([[count]])
    
    # Combine features
    final_input = hstack([text_vector, keyword_array])

    # Prediction
    prediction = model.predict(final_input)
    prob = model.predict_proba(final_input)

    confidence = round(max(prob[0]) * 100, 2)

    if prediction[0] == 1:
        return f"Fake Job ❌ ({confidence}% confidence)"
    else:
        return f"Real Job ✅ ({confidence}% confidence)"


# 🔥 ROUTES

# Home page
@app.route('/')
def home():
    return render_template("index.html")


# Prediction page (form page)
@app.route('/predict_page')
def predict_page():
    return render_template("predict.html")


# Prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['job_desc']
    result = predict_job(text)
    return render_template("predict.html", prediction=result)


# About page
@app.route('/about')
def about():
    return render_template("about.html")


# Contact page
@app.route('/contact')
def contact():
    return render_template("contact.html")


# 🔥 Run app
if __name__ == "__main__":
    app.run(debug=True)