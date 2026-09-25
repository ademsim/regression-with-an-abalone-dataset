from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Abalone Age Predictor")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "abalone_model.pkl", "rb"))


model = load_model()

st.title("Abalone Age Predictor")
st.write(
    "A Random Forest model (trained on the Kaggle Abalone dataset, Playground Series S4E4) predicts the number "
    "of shell rings of an abalone from its physical measurements. The age in years is about rings + 1.5."
)

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["Male", "Female", "Infant"])
    length = st.slider("Length (mm, scaled)", 0.05, 0.85, 0.55, 0.005)
    diameter = st.slider("Diameter (mm, scaled)", 0.05, 0.70, 0.43, 0.005)
    height = st.slider("Height (mm, scaled)", 0.0, 0.30, 0.14, 0.005)
with col2:
    whole = st.slider("Whole weight", 0.0, 3.0, 0.80, 0.01)
    shucked = st.slider("Shucked weight (meat)", 0.0, 1.5, 0.34, 0.01)
    viscera = st.slider("Viscera weight (gut)", 0.0, 0.8, 0.17, 0.01)
    shell = st.slider("Shell weight", 0.0, 1.1, 0.23, 0.01)

if st.button("Predict"):
    sex_code = {"Male": "M", "Female": "F", "Infant": "I"}[sex]
    row = {
        "Length": length,
        "Diameter": diameter,
        "Height": height,
        "Whole weight": whole,
        "Whole weight.1": shucked,
        "Whole weight.2": viscera,
        "Shell weight": shell,
        "Sex_F": 0,
        "Sex_I": 0,
        "Sex_M": 0,
    }
    row[f"Sex_{sex_code}"] = 1
    X = pd.DataFrame([row])[list(model.feature_names_in_)]

    rings = float(np.expm1(model.predict(X)[0]))
    st.success(f"Predicted rings: **{rings:.1f}**  (about {rings + 1.5:.1f} years old)")

st.caption(
    "Model: Random Forest on log-transformed rings (validation RMSLE ≈ 0.15 on the Kaggle data). "
    "Shell weight and overall size are the strongest predictors; the measurements are scaled values from the dataset."
)
