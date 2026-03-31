import json
import random

with open("chatbot/intents.json") as file:
    data = json.load(file)

def get_response(msg):
    msg = msg.lower()

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in msg:
                return random.choice(intent["responses"])

    return "Sorry, I didn't understand. Try asking about morph detection."