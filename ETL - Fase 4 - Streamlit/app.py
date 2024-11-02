import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from modules import visualizations

st.set_page_config(page_title="Dashboard de Airbnb", page_icon="🏠", layout="wide")

st.title("Dashboard de Análisis de Datos de Airbnb")
st.markdown("""
    Bienvenido al dashboard de análisis de datos de Airbnb! Aquí puedes explorar la información
    sobre las propiedades, precios y ubicaciones disponibles. Usa los filtros para personalizar tu 
    visualización.
""")

# Cargar los datos
uploader = st.file_uploader('Sube el archivo CSV de datos', type='csv')
if uploader:
    df = pd.read_csv(uploader)
    

# Barra lateral para navegar entre las pestañas
page = st.sidebar.selectbox("Navega por las secciones", ["Home", "Data Analysis", "Price Analysis", "Locations", "About"])

# Sección "Home"
if page == "Home":
    st.title("Análisis de Datos de AirBnB")
    st.write("""
    **Bienvenido al dashboard de análisis de datos de Airbnb!**  
    Explora información sobre precios, valoraciones y ubicaciones de los alojamientos.
    """)
    st.image("images/airbnb_1stpage.png", width=700)  # Ruta de la imagen en la carpeta 'images'

# Sección "Data Analysis"
elif page == "Data Analysis":
    st.header("Análisis Exploratorio de Datos")
    st.write("Gráficos y estadísticas de las propiedades de Airbnb.")

    st.subheader("Primeras Filas del DataFrame")
    st.dataframe(df.head())


    # LLamar la funciones en visualizations.py
    visualizations.rating_distribution(df) # Distribución de Ratings
    visualizations.property_type_distribution(df) # Distribución de Tipos de Propiedades
    visualizations.beds_ratings_scatter(df) # Relación entre Número de Camas y Ratings
    visualizations.beds_distribution_histogram(df) # Distribuição do Número de Camas
    visualizations.reviews_rating_distribution(df) # Distribuição de Reseñas y Valoraciones por tipo de alojamiento  


    # Agregar otros gráficos relevantes para análisis de datos (Estudiar graficos interesantes hechos en el EDA)

# Sección "Price Analysis"
elif page == "Price Analysis":
    st.header("Análisis de Precios")
    st.write("""
    En esta sección, exploraremos cómo los precios de las propiedades de Airbnb varían según el tipo de alojamiento y la ubicación. 
    Analizaremos las tendencias de precios, las distribuciones de ratings y la relación entre precios y número de reseñas. 
    Esto nos ayudará a comprender mejor el mercado de Airbnb y a identificar posibles patrones y oportunidades.
""")
    
# Agregar el control deslizante para seleccionar el rango de precios
    rango = st.slider("Selecciona un rango de precios", 
                    min_value=int(df["Prices_per_night"].min()), 
                    max_value=int(df["Prices_per_night"].max()), 
                    value=(25, 75))
    
    # Filtrar los datos por el rango de precios seleccionado
    datos_filtrados = df[(df["Prices_per_night"] > rango[0]) & (df["Prices_per_night"] < rango[1])]
    
    
    visualizations.price_property_types(df)  # Llama a la función para el gráfico por tipo de propiedad
    visualizations.price_rating_distribution(df)  # Llama a la función para la distribución de precio/rating
    visualizations.price_distribution_histogram(df)  # Distribución de Precios por Noche
    visualizations.average_price_by_capacity(df) # Precio Medio por Capacidad Máxima de Huéspedes
    visualizations.price_distribution_bar_chart(df)  # Gráfico de barras da distribuição de preços


# Sección "About"
elif page == "About":
    st.header("Sobre el Proyecto")
    st.write("""
    Este dashboard fue creado como parte de un proyecto colaborativo de análisis de datos de Airbnb.
    """)

