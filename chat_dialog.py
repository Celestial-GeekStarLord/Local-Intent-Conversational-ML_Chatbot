import numpy as np
import tensorflow as tf
import pickle

encoder_model = tf.keras.models.load_model("Model_Output/encoder_model.keras")
decoder_model = tf.keras.models.load_model("Model_Output/decoder_model.keras")

with open("Model_Output/dialog_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("Model_Output/input_len.pkl", "rb") as f:
    max_input_len = pickle.load(f)

with open("Model_Output/target_len.pkl", "rb") as f:
    max_target_len = pickle.load(f)

index_word = {v: k for k, v in tokenizer.word_index.items()}

def generate_response(input_text):
    seq = tokenizer.texts_to_sequences([input_text.lower()])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_input_len)

    states = encoder_model.predict(seq, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index["<start>"]

    decoded_sentence = []

    while True:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states, verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_word.get(sampled_token_index, "")

        if sampled_word == "<end>" or len(decoded_sentence) > max_target_len:
            break

        decoded_sentence.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states = [h, c]

    return " ".join(decoded_sentence)


print("Chatbot ready! Type quit to exit")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    response = generate_response(user_input)
    print("Bot:", response)