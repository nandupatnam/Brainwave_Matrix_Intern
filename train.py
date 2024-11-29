import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv(r"D:\BrainWave_Internship\dataset\news_datasets.csv")

# Ensure the dataset has 'text' and 'label' columns
assert "text" in data.columns and "label" in data.columns, "Dataset must contain 'text' and 'label' columns."

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model and vectorizer
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
