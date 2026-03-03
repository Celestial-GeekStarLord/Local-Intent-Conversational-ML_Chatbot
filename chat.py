import json
import numpy as np
import tensorflow as tf
import pickle
import random

# Load files
model = tf.keras.models.load_model("Intent_Model/chat_model.keras")

with open("Intent_Model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("Intent_Model/label_encoder.pkl", "rb") as f:
    lbl_encoder = pickle.load(f)

with open("intents.json") as file:
    data = json.load(file)

max_len = 20

print("Chatbot ready! (type quit to stop)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    seq = tokenizer.texts_to_sequences([user_input])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded, verbose=0)
    tag = lbl_encoder.inverse_transform([np.argmax(pred)])[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            print("Bot:", random.choice(intent["responses"]))