import streamlit as st
from src.predict import predict

st.title("Fake News Detector with BERT")

text = st.text_area("Enter a news article or headline:")

if st.button("Predict"):
    label = predict(text)
    st.write(f"### This article is: {label}")
