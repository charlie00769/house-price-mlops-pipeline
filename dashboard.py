import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

st.title("🏠 House Price Prediction")

area = st.number_input("Enter Area (sq ft)")
bedrooms = st.number_input("Enter Bedrooms")

if st.button("Predict"):
    data = np.array([[area, bedrooms]])
    price = model.predict(data)[0]
    st.success(f"Estimated Price: ₹ {price:,.2f}")