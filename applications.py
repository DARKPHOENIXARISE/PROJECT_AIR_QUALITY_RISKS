# app.py — compact & fixed (encode City, scale only numeric cols, align features)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")  # keep UI clean (sklearn version warnings)

st.set_page_config(page_title="Air Quality Prediction")
st.title("Air Quality Prediction")

# ----- helpers to load artifacts -----
def load_required(path, name):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load required {name} from '{path}': {e}")
        st.stop()

def load_optional(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

# ----- load model + artifacts -----
model = load_required("rf_model_best.pkl", "model")
scaler = load_optional("scaler.pkl")           # optional
city_encoder = load_optional("city_encoder.pkl")
label_encoder = load_optional("label_encoder.pkl")  # optional target decoder

# ----- expected fallback order (if model has no feature_names_in_) -----
fallback_order = ['City','PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene']

# ----- city options for selectbox -----
default_cities = ["Ahmedabad","Mumbai","Delhi","Kolkata","Bengaluru","Aizawl"]
if city_encoder is not None and hasattr(city_encoder, "classes_"):
    city_options = list(city_encoder.classes_)
else:
    city_options = default_cities
    if city_encoder is None:
        st.sidebar.warning("city_encoder.pkl not found — selectbox uses default list (predictions may fail if the model expects encoded cities).")

# ----- user input UI -----
def get_user_input():
    city = st.sidebar.selectbox("City", city_options)
    pm25 = st.sidebar.number_input("PM2.5", value=50.0, format="%.2f")
    pm10 = st.sidebar.number_input("PM10", value=80.0, format="%.2f")
    no = st.sidebar.number_input("NO", value=10.0, format="%.2f")
    no2 = st.sidebar.number_input("NO2", value=20.0, format="%.2f")
    nox = st.sidebar.number_input("NOx", value=15.0, format="%.2f")
    nh3 = st.sidebar.number_input("NH3", value=5.0, format="%.2f")
    co = st.sidebar.number_input("CO", value=1.0, format="%.2f")
    so2 = st.sidebar.number_input("SO2", value=10.0, format="%.2f")
    o3 = st.sidebar.number_input("O3", value=30.0, format="%.2f")
    benzene = st.sidebar.number_input("Benzene", value=1.0, format="%.2f")
    toluene = st.sidebar.number_input("Toluene", value=2.0, format="%.2f")

    return pd.DataFrame([{
        "City": city,
        "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2, "NOx": nox,
        "NH3": nh3, "CO": co, "SO2": so2, "O3": o3, "Benzene": benzene, "Toluene": toluene
    }])

input_df = get_user_input()
with st.expander("Input preview (raw)"):
    st.write(input_df)

# ----- prediction -----
if st.button("Predict AQI Bucket"):
    # copy to avoid mutating original
    df = input_df.copy()

    # 1) encode City (required if model trained on encoded city)
    if "City" in df.columns:
        if city_encoder is None:
            st.error("city_encoder.pkl missing — model likely expects encoded city. Provide city_encoder.pkl used at training.")
            st.stop()
        try:
            df["City"] = city_encoder.transform(df["City"].astype(str))
        except Exception as e:
            st.error(f"City encoding failed: {e}")
            st.write("Known cities in encoder:", list(city_encoder.classes_))
            st.stop()

    # 2) determine expected column order
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    else:
        expected_cols = fallback_order

    # 3) add missing expected columns with zeros (safe fallback), then reorder
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[expected_cols]  # reorder

    # 4) scale only numeric columns that the scaler was fit on
    X_proc = df.copy()
    if scaler is not None:
        # treat City as categorical (do not include City in numeric_cols_for_scaler)
        numeric_cols_for_scaler = [c for c in expected_cols if c != "City"]
        # if scaler was fit on a different set, attempting transform will raise; handle gracefully
        try:
            # transform only the numeric subset (safe — avoids passing 'City' to scaler)
            scaled_vals = scaler.transform(X_proc[numeric_cols_for_scaler])
            X_proc[numeric_cols_for_scaler] = scaled_vals
        except Exception as e:
            # fallback: do not scale (but warn). This avoids the "unseen feature 'City'" failure.
            st.warning(f"Scaler transform failed, proceeding without scaling: {e}")

    # 5) final shape check
    X_arr = X_proc.values
    if hasattr(model, "n_features_in_") and X_arr.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch: model expects {model.n_features_in_} features but got {X_arr.shape[1]}.")
        st.write("Model expected order:", expected_cols)
        st.write("Provided columns:", df.columns.tolist())
        st.stop()

    # 6) predict
    try:
        preds = model.predict(X_arr)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # 7) decode label or use fallback mapping
    if label_encoder is not None:
        try:
            label_out = label_encoder.inverse_transform(preds.astype(int))[0]
        except Exception:
            label_out = str(preds[0])
    else:
        fallback_labels = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
        try:
            label_out = fallback_labels[int(preds[0])]
        except Exception:
            label_out = str(preds[0])

    st.subheader("Prediction Result")
    st.success(f"Predicted AQI Bucket: {label_out}")

    # debug info for verification
    with st.expander("Debug: processed input → model"):
        st.write("Processed DataFrame (after encoding/reorder/scale if applied):")
        st.write(X_proc)
        st.write("dtypes:")
        st.write(X_proc.dtypes)
        if hasattr(model, "feature_names_in_"):
            st.write("model.feature_names_in_:", list(model.feature_names_in_))
        st.write("Array fed to model:")
        st.write(X_arr)
