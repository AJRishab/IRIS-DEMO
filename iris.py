# iris_streamlit_advanced.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('iris_classifier_model.pkl')
scaler = joblib.load('iris_scaler.pkl')

# Class names
target_names = ['setosa', 'versicolor', 'virginica']

# Set page config
st.set_page_config(page_title="Iris Classifier 🌸", page_icon="🌸", layout="centered")

# Sidebar
st.sidebar.title("🌟 About the App")
st.sidebar.write("""
This app uses a Machine Learning model (SVM) to classify 
Iris flowers into three species:
- Setosa
- Versicolor
- Virginica

Based on sepal and petal measurements.
""")
st.sidebar.markdown("---")
st.sidebar.write("**Made by [AJRishab]**")

# Main title
st.title("🌸 Iris Flower Species Classifier")
st.write("Enter the flower measurements below to predict the Iris species:")

# Input fields
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button("🌟 Predict"):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    
    prediction = model.predict(sample_scaled)
    probabilities = model.predict_proba(sample_scaled)[0]

    predicted_species = target_names[prediction[0]]
    confidence = probabilities[prediction[0]] * 100

    st.success(f"🌸 Predicted Species: **{predicted_species.capitalize()}**")
    st.info(f"🔍 Confidence: **{confidence:.2f}%**")

    # Bar chart of probabilities
    st.subheader("📊 Prediction Probabilities:")

    fig, ax = plt.subplots()
    species = target_names
    probs = probabilities
    colors = ['#a1dab4', '#41b6c4', '#225ea8']

    ax.bar(species, probs, color=colors)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Confidence for Each Species')

    # Annotate bars
    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    st.pyplot(fig)

    st.balloons()

# Footer
st.markdown("---")
st.caption("**Disclaimer:** This is a demo application for educational purposes. The model's performance may vary based on the input data.")
