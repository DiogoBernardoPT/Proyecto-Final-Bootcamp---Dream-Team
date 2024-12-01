import streamlit as st
import joblib
import numpy as np
import pandas as pd
from modules.visualizations import show_feature_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# Funciones auxiliares para cargar los diferentes modelos y escaladores
@st.cache_resource
def load_scalers():
    x_scaler = joblib.load('models/x_scaler.pkl')
    y_scaler = joblib.load('models/y_scaler.pkl')
    return x_scaler, y_scaler

@st.cache_resource
def load_models():
    return {
        "lightgbm": joblib.load("models/best_lightgbm_model.pkl"),
        "neural_network": joblib.load("models/simple_nn_model.pkl"),
        "recommendation": joblib.load("models/NearestNeighbors.pkl"),
    }

def show_price_prediction(df_processed):
    st.header("💸 Price Prediction Model")
    st.write("Using this LightGBM model, we will predict the price of a property based on its features.")

    # Cargar los modelos y el KNN imputer
    models = load_models()
    
    model = models["lightgbm"]["model"]
    
    # Cargar el KNN Imputer
    knn_imputer = joblib.load("models/knn_imputer.pkl")

    # Entradas del usuario
    st.subheader("Please enter the following details about the property:")

    max_guests = st.sidebar.slider("Maximum Guests", min_value=1, max_value=10, value=4)
    bedrooms = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=10, value=2)
    beds = st.sidebar.slider("Number of Beds", min_value=1, max_value=10, value=2)
    bathrooms = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=5, value=1)
    cleaning_fee = st.sidebar.slider("Cleaning Fee (€)", min_value=0, value=50)

    # Crear un DataFrame con las entradas del usuario
    user_input = {
        "maximum_guests": max_guests,
        "dormitorios": bedrooms,
        "camas": beds,
        "baños": bathrooms,
        "cleaning_fee": cleaning_fee
    }

    input_df = pd.DataFrame(user_input, index=[0])

    # Rellenar las columnas restantes con el KNN Imputer (sin incluir 'prices_per_night')
    columns_to_impute = [col for col in df_processed.columns if col not in user_input and col != "prices_per_night"]

    for col in columns_to_impute:
        input_df[col] = np.nan

    # Alinear las columnas del input_df con las que se utilizaron para entrenar el modelo
    expected_columns = df_processed.drop(columns=["prices_per_night"]).columns

    # Agregar columnas faltantes si es necesario
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[expected_columns]  # Asegurar el orden correcto

    # Imputar valores faltantes
    imputed_values = knn_imputer.transform(input_df)
    for i, col in enumerate(columns_to_impute):
        input_df[col] = imputed_values[0][i]

    # Cargar el escalador
    x_scaler, y_scaler = load_scalers()

    # Escalar las características
    features_scaled = x_scaler.transform(input_df)

    # Realizar la predicción
    predicted_price_scaled = model.predict(features_scaled)[0]
    predicted_price = y_scaler.inverse_transform([[predicted_price_scaled]])[0][0]

    # Mostrar la predicción
    st.write(f"**The predicted price for this property is: €{predicted_price:.2f}**.")

    st.subheader("Model Evaluation Metrics")
    metrics_df = pd.read_csv("data/lightgbm_metrics.csv")

    st.dataframe(metrics_df)

    with st.expander("Feature Importance Chart"):
    # Call the visualization function
        show_feature_importance()
    
    # Write the hypothesis below the chart
        st.write("""
        - **Ratings**: Properties with higher ratings are likely to command premium prices, reflecting customer satisfaction.
        - **Number of Reviews**: A higher number of reviews often indicates popularity and demand.
        - **Cleaning Fee**: The additional cleaning fee impacts the total price, highlighting its importance in determining overall cost.
        - **Kitchen and Dining Amenities**: These features contribute significantly to pricing, as they add value for longer stays or family-oriented accommodations.
        - **Exterior Features**: Outdoor spaces or aesthetics enhance property appeal and justify higher prices.
        These factors align with the intuition that customer feedback, service fees, and amenities are key determinants of property pricing in the Airbnb market.
    """)
    

def show_neural_network_price_prediction(df_processed):
    st.header("💸 Price Prediction Model (Neural Networks)")

    models = load_models()
    model = models["neural_network"]

    knn_imputer = joblib.load("models/knn_imputer.pkl")

    st.subheader("Please enter the following details about the property:")

    max_guests = st.sidebar.slider("Maximum Guests", min_value=1, max_value=10, value=4)
    bedrooms = st.sidebar.slider("Number of Bedrooms", min_value=1, max_value=10, value=2)
    beds = st.sidebar.slider("Number of Beds", min_value=1, max_value=10, value=2)
    bathrooms = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=5, value=1)
    cleaning_fee = st.sidebar.slider("Cleaning Fee (€)", min_value=0, value=50)

    user_input = {
        "maximum_guests": max_guests,
        "dormitorios": bedrooms,
        "camas": beds,
        "baños": bathrooms,
        "cleaning_fee": cleaning_fee
    }

    input_df = pd.DataFrame(user_input, index=[0])

    columns_to_impute = [col for col in df_processed.columns if col not in user_input and col != "prices_per_night"]

    for col in columns_to_impute:
        input_df[col] = np.nan

    expected_columns = df_processed.drop(columns=["prices_per_night"]).columns

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[expected_columns]

    imputed_values = knn_imputer.transform(input_df)
    for i, col in enumerate(columns_to_impute):
        input_df[col] = imputed_values[0][i]

    x_scaler, y_scaler = load_scalers()

    # Escalonamento das entradas
    features_scaled = x_scaler.transform(input_df)

    # Predicciones con los datos escalados
    predicted_price_scaled = model.predict(features_scaled)

    # Desescalonamento do preço previsto
    predicted_price = y_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]

    st.write(f"**The predicted price for this property is: €{predicted_price:.2f}**.")

    st.subheader("Model Evaluation Metrics")
    metrics_df = pd.read_csv("data/simple_nn_metrics.csv") 

    st.dataframe(metrics_df)

    st.subheader("Model Training and Validation Loss")
    train_val_loss_fig = joblib.load("images/analysis/train_val_loss.pkl")
    st.plotly_chart(train_val_loss_fig)

    st.subheader("Real vs Predicted Prices")
    real_vs_pred_fig = joblib.load("images/analysis/real_vs_pred.pkl")
    st.plotly_chart(real_vs_pred_fig)


def show_recommender_and_nlp(df_processed, df_sentiment):
    st.header("🏘️ Recommender System + Sentiment Analysis")
    st.write("Select an Airbnb to view sentiment analysis and recommendations.")
    
def show_model_explanation(model_choice):
    st.subheader("Model Explanation")

    if model_choice == "Price Prediction - LightGBM":
        st.write("""
        The LightGBM model is a gradient boosting algorithm that predicts property prices based on features such as the number of bedrooms, cleaning fee, and other relevant characteristics.
        This model was trained using a dataset of Airbnb listings, where we carefully tuned its hyperparameters to achieve optimal predictive performance.
        """)
        st.write("""
        To ensure we used the best model for price prediction, we tested several algorithms, including Linear Regression, Random Forest, Gradient Boosting, XGBoost, and MLP Regressor. 
        After evaluating their performance based on their metrics, we found LightGBM to deliver the best balance between accuracy and efficiency.
        """)
        st.markdown("### Comparison of Model Results")
        resultados_modelos = pd.read_pickle("models/resultados_modelos.pkl")
        st.dataframe(resultados_modelos)        

    elif model_choice == "Price Prediction - Neural Networks":
        st.write("""
            The neural network model is based on a deep learning approach, where multiple layers of neurons are trained to predict property prices.
            The network is designed to learn from complex patterns in the data, including interactions between features such as the number of bedrooms, cleaning fee, and other aspects.
        """)
        
    elif model_choice == "Recommender + NLP Sentiment Analysis":
        st.write("""
            The recommendation system suggests Airbnb properties based on user preferences, such as the number of guests, bedrooms, and other attributes.
            The sentiment analysis part analyzes the reviews of the properties to determine whether guests had positive or negative experiences.
            This can help users make informed decisions when selecting a property.
        """)
        
        st.write("Sentiment analysis uses NLP techniques to analyze review text and classify sentiment.")

# Función principal de visualización en la página
def show():
    # Recuperando os dados processados
    df_processed = st.session_state.df_processed
    df_sentiment = st.session_state.df_sentiment

    # Barra lateral para elegir el modelo
    model_choice = st.sidebar.selectbox(
        "Choose the analysis model:",
        ["Price Prediction - LightGBM", "Price Prediction - Neural Networks", "Recommender + NLP Sentiment Analysis"]
    )

    # Mostrar la explicación del modelo seleccionado
    show_model_explanation(model_choice)

    # Mostrar los resultados según la elección del modelo
    if model_choice == "Price Prediction - LightGBM":
        show_price_prediction(df_processed)
    elif model_choice == "Price Prediction - Neural Networks":
        show_neural_network_price_prediction(df_processed)
    elif model_choice == "Recommender + NLP Sentiment Analysis":
        show_recommender_and_nlp(df_processed, df_sentiment)