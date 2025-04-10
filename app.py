from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data.get('news_text', '')

    if not news_text.strip():
        return jsonify({'prediction': 'Invalid input. Please provide some text.'})

    input_vector = vectorizer.transform([news_text])

    prediction = model.predict(input_vector)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
