
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Title
st.title("Sylheti Sentiment Analysis - CNN-LSTM Model")

# Load dataset (optional - for info)
df = pd.read_excel("27.04.2025_wrong_dataset.xlsx")

# Display some example data
if st.checkbox("Show sample dataset"):
    st.write(df.head())

# Load tokenizer and model
try:
    tokenizer = joblib.load("tokenizer.pkl")
    model = tf.keras.models.load_model("sentiment_model.h5")
except Exception as e:
    st.error("Error loading model or tokenizer. Please check deployment files.")
    st.stop()

# Input box
user_input = st.text_area("Enter Sylheti text here for sentiment prediction:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=100)
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction)

        labels = ["Positive", "Negative", "Neutral"]
        st.success(f"Predicted Sentiment: **{labels[predicted_class]}**")
