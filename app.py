
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diamond Type Prediction",
    page_icon="💎",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load Model and Columns
# -----------------------------
model = joblib.load(
    os.path.join(BASE_DIR, "diamond_model.pkl")
)

columns = joblib.load(
    os.path.join(BASE_DIR, "diamond_columns.pkl")
)

# -----------------------------
# App Title
# -----------------------------
st.title("💎 Diamond Type Classification")
st.write("Enter diamond details to predict its type.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    carat = st.number_input(
        "Carat",
        min_value=0.01,
        value=1.0,
        step=0.01
    )

    cut = st.selectbox(
        "Cut",
        ["Fair", "Good", "Ideal", "Super Ideal", "Very Good"]
    )

    shape = st.selectbox(
        "Shape",
        [
            "Asscher", "Cushion", "Emerald", "Heart",
            "Marquise", "Oval", "Pear", "Princess",
            "Radiant", "Round"
        ]
    )

with col2:
    date_fetched = st.date_input("Date Fetched")

    # Automatically detect type classes from encoded columns
    type_columns = [
        col for col in columns
        if col.startswith("type_")
    ]

    type_options = [
        col.replace("type_", "")
        for col in type_columns
    ]

    st.info(
        "The model will predict the diamond type "
        "based on the entered features."
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Diamond Type", type="primary"):

    # Create input row in the same column order
    input_data = pd.DataFrame(
        np.zeros((1, len(columns))),
        columns=columns
    )

    # Numerical feature
    if "carat" in input_data.columns:
        input_data["carat"] = carat

    # Cut encoding
    cut_column = f"cut_{cut}"
    if cut_column in input_data.columns:
        input_data[cut_column] = 1

    # Shape encoding
    shape_column = f"shape_{shape}"
    if shape_column in input_data.columns:
        input_data[shape_column] = 1

    # Date encoding
    if "date_fetched" in input_data.columns:
        input_data["date_fetched"] = pd.Timestamp(
            date_fetched
        ).toordinal()

    # -----------------------------
    # Apply scaler if your model
    # was trained using scaled data
    # -----------------------------
    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction Completed!")

        st.subheader("💎 Predicted Diamond Type")

        st.markdown(f"## {prediction}")

        

    except Exception as e:
        st.error(f"Prediction Error: {e}")
