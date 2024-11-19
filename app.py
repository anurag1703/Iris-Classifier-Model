import streamlit as st
import numpy as np
import joblib

# Load the trained model, scaler, and label encoder
model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")  # Load the saved LabelEncoder

# Streamlit App Title
st.title("Iris Flower Classification")

# App Description
st.write("""
This application predicts the species of an Iris flower based on the following features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
""")

# User Inputs for Features
sepal_length = st.number_input("Enter Sepal Length (in cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Enter Sepal Width (in cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Enter Petal Length (in cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Enter Petal Width (in cm)", min_value=0.0, step=0.1)

# Button for Prediction
if st.button("Predict"):
    try:
        # Prepare input features for prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)  # Scale the input features
        
        # Make Prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)

        # Convert numeric prediction to flower name
        predicted_species = label_encoder.inverse_transform(prediction)[0]

        # Display Results
        st.write(f"The predicted species is: **{predicted_species}**")
        st.write("Prediction probabilities:")
        for i, prob in enumerate(prediction_proba[0]):
            species_name = label_encoder.inverse_transform([i])[0]  # Get flower name
            st.write(f"{species_name}: {prob:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
