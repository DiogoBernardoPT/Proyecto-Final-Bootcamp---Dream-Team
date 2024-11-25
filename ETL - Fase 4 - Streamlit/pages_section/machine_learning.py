import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# Funciones auxiliares para cargar los diferentes modelos y escaladores
@st.cache_resource
def load_scalers():
    x_scaler = joblib.load('models/x_scaler.pkl')  # Carregar x_scaler
    y_scaler = joblib.load('models/y_scaler.pkl')  # Carregar y_scaler
    return x_scaler, y_scaler
    

@st.cache_resource
def load_lightgbm_model():
    return joblib.load("models/best_lightgbm_model.pkl")

@st.cache_resource
def load_nlp_model():
    return joblib.load("models/simple_nn_model.pkl")

@st.cache_resource
def load_recommendation_model():
    return joblib.load("models/NearestNeighbors.pkl")


# Funciones para mostrar cada tipo de análisis
def show_price_prediction(df_processed):
    st.header("💸 Price Prediction Model")
    st.write("Using this LightGBM model, we will predict the price of a property based on its features.")

    model_data = load_lightgbm_model()
    model = model_data["model"]
    original_columns = model_data["original_columns"]

    # Linea base del DataFrame procesado
    default_row = df_processed[original_columns].iloc[0].copy()

    # Inputs para el usuário
    st.subheader("Please enter the following details about the property:")

    # Selección de variables del usuario
    max_guests = st.slider("Maximum Guests", min_value=1, max_value=10, value=4)
    bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=2)
    beds = st.slider("Number of Beds", min_value=1, max_value=10, value=2)
    bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=5, value=1)
    cleaning_fee = st.number_input("Cleaning Fee (€)", min_value=0, value=50)
    rating = st.slider("Rating", min_value=1, max_value=5, value=4)

    default_row["max_guests"] = max_guests
    default_row["bedrooms"] = bedrooms
    default_row["beds"] = beds
    default_row["bathrooms"] = bathrooms
    default_row["cleaning_fee"] = cleaning_fee
    default_row["rating"] = rating
    
    # Crear el vector de características basado en las entradas del usuario
    features = default_row[original_columns].values.reshape(1, -1)

    st.write("Features being passed to the model:", features)

    x_scaler, y_scaler = load_scalers()

    features_scaled = x_scaler.transform(features)

    predicted_price_scaled = model.predict(features_scaled)[0]

    predicted_price = y_scaler.inverse_transform([[predicted_price_scaled]])[0][0]

    # Mostrar a previsão
    st.write(f"**The predicted price for this property is: €{predicted_price:.2f}**.")

    # Mostrar las métricas del modelo
    st.subheader("Model Evaluation Metrics")


def show_recommender(df_processed): # Cambiar aqui el nombre si se guarda con otro nombre - actualizar codigo code neighbours
    st.header("🏘️ Recommender System")
    st.write("Displaying recommendations...")
    # Resto del codigo aqui


def show_nlp_analysis(df_sentiment):
    st.header("📝 NLP - Sentiment Analysis")
    st.write("Displaying sentiment analysis...")
    


# Función principal de la página Machine Learning
def show():

    df_processed = st.session_state.df_processed
    df_sentiment = st.session_state.df_sentiment

    analysis_type = st.selectbox(
        "Select the analysis you want to perform:",
        ["Price Prediction", "Recommender System", "NLP Sentiment Analysis"]
    )

    if analysis_type == "Price Prediction":
        show_price_prediction(df_processed)
    elif analysis_type == "Recommender System":
        show_recommender(df_processed)
    elif analysis_type == "NLP Sentiment Analysis":
        show_nlp_analysis(df_sentiment)