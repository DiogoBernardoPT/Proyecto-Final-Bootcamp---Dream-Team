import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(page_title="Dashboard de Airbnb", page_icon="🏠", layout="wide")

st.title("Dashboard de Análisis de Datos de Airbnb")
st.markdown("""
    Bienvenido al dashboard de análisis de datos de Airbnb! Aquí puedes explorar la información
    sobre las propiedades, precios y ubicaciones disponibles. Usa los filtros para personalizar tu 
    visualización.
""")
# Control deslizante para seleccionar rango
rango = st.slider("Selecciona un rango", 0, 100, (25, 75))
# Cargar los datos
datos = pd.read_csv('C:/Users/peni_/Desktop/proyecto/Proyecto-Final-Bootcamp---Dream-Team/df_clean.csv')
# Filtrar los datos por el rango seleccionado
datos_filtrados = datos[(datos["Prices_per_night"] > rango[0]) & (datos["Prices_per_night"] < rango[1])]
st.write(datos.head())  # Muestra las primeras filas para confirmar la carga
# Visualización de gráfico seleccionando solo la columna de precios
st.line_chart(datos_filtrados["Prices_per_night"])
