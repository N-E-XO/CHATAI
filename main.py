import pickle

# Load the trained AI model
with open("sid_model.pkl", "rb") as file:
    model = pickle.load(file)

# ChatAI's responses
responses = {
    "greeting": "Hello! Welcome to ChatAI!",
    "identity": "I am ChatAI, your personal AI chatbot.",
    "creator": "I was created by my developer.",
    "programming": "Python is a programming language used to build software and AI.",
    "ai": "AI allows computers to perform tasks that usually require human intelligence.",
    "goodbye": "Goodbye! Have a great day!"
}

print("ChatAI: Hello! I'm ChatAI. Let's talk!")

while True:
    question = input("You: ").lower().strip()

    if question == "exit":
        print("ChatAI: Goodbye!")
        break

    if question == "":
        continue

    probabilities = model.predict_proba([question])[0]

    best_index = probabilities.argmax()
    confidence = probabilities[best_index]

    intent = model.named_steps["classifier"].classes_[best_index]
    print("DEBUG:", intent, confidence)

    if confidence < 0.35:
        print("ChatAI: I don't know that yet. Teach me!")
    else:
        print("ChatAI:", responses[intent])
