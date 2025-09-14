# app.py

# ==== Import required libraries ====
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==== Load required models and encoders ====
model = joblib.load("rf_model_best.pkl")
scaler = joblib.load("scaler.pkl")         # if you scaled features before training
city_encoder = joblib.load("city_encoder.pkl")  # LabelEncoder for city
label_encoder = joblib.load("label_encoder.pkl")

# ==== Page config ====
st.set_page_config(page_title="Air Quality Prediction")
st.title("Air Quality Prediction App")
st.markdown("Predict **Air Quality Index (AQI Bucket)** based on environmental factors.")

# ==== Sidebar inputs ====
st.sidebar.header("Input Parameters")

def get_user_input():
    # Example parameters (you can add/remove based on your dataset features)
    city = st.sidebar.selectbox("City", city_encoder.classes_)
    pm25 = st.sidebar.number_input("PM2.5", min_value=0.0, max_value=500.0, value=50.0)
    pm10 = st.sidebar.number_input("PM10", min_value=0.0, max_value=600.0, value=80.0)
    no = st.sidebar.number_input("NO", min_value=0.0, max_value=200.0, value=10.0)
    no2 = st.sidebar.number_input("NO2", min_value=0.0, max_value=200.0, value=20.0)
    nox = st.sidebar.number_input("NOx", min_value=0.0, max_value=200.0, value=15.0)
    nh3 = st.sidebar.number_input("NH3", min_value=0.0, max_value=200.0, value=5.0)
    co = st.sidebar.number_input("CO", min_value=0.0, max_value=10.0, value=1.0)
    so2 = st.sidebar.number_input("SO2", min_value=0.0, max_value=200.0, value=10.0)
    o3 = st.sidebar.number_input("O3", min_value=0.0, max_value=200.0, value=30.0)
    benzene = st.sidebar.number_input("Benzene", min_value=0.0, max_value=50.0, value=1.0)
    toluene = st.sidebar.number_input("Toluene", min_value=0.0, max_value=100.0, value=2.0)




    data = {
        'City': city,
        'PM2.5': pm25,
        'PM10': pm10,
        'NO': no,
        'NO2': no2,
        'NOx': nox,
        'NH3': nh3,
        'CO': co,
        'SO2': so2,
        'O3': o3,
        'Benzene': benzene,
        'Toluene': toluene
    }

    return pd.DataFrame([data])

# Get user input
input_df = get_user_input()

# ==== Prediction button ====
if st.button("Predict AQI Bucket"):
    # Scale input if scaler was used
    # input_scaled = scaler.transform(input_df)
    # prediction = model.predict(input_scaled)

    prediction = model.predict(input_df)

    st.subheader("Prediction Result")
    results = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']   # Or your AQI bucket categories
    st.success(f"**Predicted AQI Bucket:** '{results[prediction[0]]}'")
