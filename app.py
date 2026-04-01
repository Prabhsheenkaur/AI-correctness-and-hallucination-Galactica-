import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Wine Quality Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved files
model = joblib.load(os.path.join(BASE_DIR, "best_wine_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "wine_scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "wine_feature_names.pkl"))

# Load model results
results_path = os.path.join(BASE_DIR, "model_results.csv")
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
else:
    results_df = pd.DataFrame()

# Load dataset
red = pd.read_csv(os.path.join(BASE_DIR, "winequality-red.csv"), sep=";")
white = pd.read_csv(os.path.join(BASE_DIR, "winequality-white.csv"), sep=";")

red["wine_type"] = "red"
white["wine_type"] = "white"
df = pd.concat([red, white], ignore_index=True)

menu = st.sidebar.selectbox(
    "Select Page",
    [
        "Dataset Overview",
        "Quality Distribution",
        "Feature Importance",
        "Correlation Matrix",
        "Model Results",
        "Prediction"
    ]
)

st.title("Wine Quality Prediction Dashboard")

if menu == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.dataframe(df.head())
    st.dataframe(df.describe())

elif menu == "Quality Distribution":
    st.subheader("Wine Quality Distribution")
    counts = df["quality"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Wine Quality Distribution")
    ax.set_xlabel("Quality")
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif menu == "Feature Importance":
    st.subheader("Feature Importance - Random Forest")

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

elif menu == "Correlation Matrix":
    st.subheader("Correlation Matrix")

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, interpolation="nearest")
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Matrix")

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)

elif menu == "Model Results":
    st.subheader("Model Performance Comparison")

    if results_df.empty:
        st.warning("model_results.csv not found. Please save your model comparison results first.")
    else:
        st.dataframe(results_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(results_df["Model"], results_df["Accuracy"])
        ax.set_title("Accuracy by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.bar(results_df["Model"], results_df["Weighted_F1"])
        ax2.set_title("Weighted F1 Score by Model")
        ax2.set_xlabel("Model")
        ax2.set_ylabel("Weighted F1")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

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