from flask import Flask, request, jsonify, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

@app.route('/')
def index():
    # Render the main HTML file
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    news_text = data.get('news_text', '')

    if not news_text.strip():
        return jsonify({'prediction': 'Invalid input. Please provide some text.'})

    # Vectorize the input text
    input_vector = vectorizer.transform([news_text])

    # Make a prediction
    prediction = model.predict(input_vector)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
