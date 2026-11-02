
import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load training data
with open("training_data.json", "r") as file:
    data = json.load(file)

# Separate sentences and categories
sentences = [item["text"] for item in data]
labels = [item["intent"] for item in data]

# Create the AI model
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(sentences, labels)

# Save the trained model
with open("sid_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("ChatAI trained successfully!")
