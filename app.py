import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Quality Dashboard", layout="wide")

model = joblib.load("best_wine_model.pkl")
scaler = joblib.load("wine_scaler.pkl")
feature_names = joblib.load("wine_feature_names.pkl")
results_df = pd.read_csv("model_results.csv")

red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")   
red["wine_type"] = "red"
white["wine_type"] = "white"
df = pd.concat([red, white], ignore_index=True)

st.title("Wine Quality Prediction Dashboard")

menu = st.sidebar.selectbox(
    "Select Page",
    ["Dataset Overview", "Model Results", "Prediction"]
)

if menu == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.dataframe(df.head())

    st.subheader("Quality Distribution")
    fig, ax = plt.subplots()
    counts = df["quality"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Quality")
    ax.set_ylabel("Count")
    ax.set_title("Wine Quality Distribution")
    st.pyplot(fig)

elif menu == "Model Results":
    st.subheader("Model Comparison")
    st.dataframe(results_df)

    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["Weighted_F1"])
    ax.set_title("Weighted F1 Score by Model")
    ax.set_ylabel("Weighted F1")
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif menu == "Prediction":
    st.subheader("Predict Wine Quality")

    input_data = {}

    defaults = {
        "fixed acidity": 7.0,
        "volatile acidity": 0.3,
        "citric acid": 0.3,
        "residual sugar": 6.0,
        "chlorides": 0.05,
        "free sulfur dioxide": 30.0,
        "total sulfur dioxide": 115.0,
        "density": 0.995,
        "pH": 3.2,
        "sulphates": 0.5,
        "alcohol": 10.5,
        "wine_type": 0
    }

    for feature in feature_names:
        if feature == "wine_type":
            wine_type = st.selectbox("Wine Type", ["red", "white"])
            input_data[feature] = 0 if wine_type == "red" else 1
        else:
            input_data[feature] = st.number_input(
                feature,
                value=float(defaults.get(feature, 0.0)),
                format="%.4f"
            )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted Wine Quality: {prediction}")