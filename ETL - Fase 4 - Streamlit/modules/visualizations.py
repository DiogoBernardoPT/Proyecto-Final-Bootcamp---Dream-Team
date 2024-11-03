import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Función para el gráfico de distribución de valoraciones
def rating_distribution(df):
    st.subheader("Distribución de Ratings")
    rating_hist = px.histogram(df, x="Ratings", nbins=20, title="Distribución de Ratings")
    st.plotly_chart(rating_hist, use_container_width=True, key="rating_hist")


# Función para el gráfico de precio por tipo de propiedad
def price_property_types(df):
    # st.header("Precio por Tipo de Propiedad")
    price_chart = px.box(df, x="Property_types", y="Prices_per_night", title="Precios por Tipo de Alojamiento")
    st.plotly_chart(price_chart, use_container_width=True, key="price_property_types")

# Función para la gráfica de visualización entre Precio/Ratings por tipo de alojamiento
def price_rating_distribution(df):
    price_rating_distr_log = px.scatter(
        data_frame=df,
        x="Ratings",
        y="Prices_per_night",
        log_x=True,
        color="Property_types",
        title="Distribución de Precio y Valoración por Tipo de Alojamiento (Escala Logarítmica)",
        labels={"Ratings": "Valoración (escala logarítmica)", "Prices_per_night": "Precio por Noche"},
    )
    st.plotly_chart(price_rating_distr_log, use_container_width=True)

# Función para la gráfica de precio medio por capacidad máxima de huéspedes
def average_price_by_capacity(df):
    # Gráfico de barras que ilustra el precio medio según la capacidad máxima de huéspedes.
    avg_price = df.groupby('Maximum_guests')['Prices_per_night'].mean().reset_index()
    fig = px.bar(avg_price, x='Maximum_guests', y='Prices_per_night', 
                title="Precio Medio por Capacidad Máxima de Huéspedes")
    st.plotly_chart(fig, use_container_width=True)    

# Función para la gráfica de histograma que muestra la frecuencia de precios en los alojamientos
def price_distribution_histogram(df):
    fig = px.histogram(df, x='Prices_per_night', nbins=30, 
                    title="Distribución de Precios por Noche")
    st.plotly_chart(fig, use_container_width=True)

# Función para el gráfico de barras que agrupa precios en intervalos para analizar su distribución
def price_distribution_bar_chart(df):
    # Agrupar por intervalos de preço
    intervalos = pd.cut(df["Prices_per_night"], bins=10)
    conteo_por_intervalo = df.groupby(intervalos).size()

    bar_chart = px.bar(x=conteo_por_intervalo.index.astype(str), 
                        y=conteo_por_intervalo.values, 
                        labels={'x': 'Intervalos de Precio', 'y': 'Conteo'},
                        title='Distribución de Precios en el Rango Seleccionado')
    st.plotly_chart(bar_chart, use_container_width=True)

# Función para la gráfica de distribución de Reseñas y Valoraciones por tipo de alojamiento
def reviews_rating_distribution(df):
    reviews_rating_distr = px.scatter(
        data_frame=df,
        x="Ratings",
        y="Num_reviews",
        log_x=True,
        color="Prices_per_night",
        title="Distribución de Reseñas y Valoración por Tipo de Alojamiento (Escala Logarítmica)",
        hover_name="Property_types",
        opacity=0.5,
        size="Prices_per_night",
        size_max=15,
    )
    st.plotly_chart(reviews_rating_distr, use_container_width=True)

# Función para la gráfica de distribución de tipos de propiedades
def property_type_distribution(df):
    property_count = df['Property_types'].value_counts()
    fig = px.pie(property_count, values=property_count.values, names=property_count.index, 
                title="Distribución de Tipos de Propiedades")
    st.plotly_chart(fig, use_container_width=True)

# Función para la gráfica de histograma de número de camas
def beds_distribution_histogram(df):
    fig = px.histogram(df, x='Camas', nbins=30, 
                    title="Distribución del Número de Camas")
    st.plotly_chart(fig, use_container_width=True)

# Función para la gráfica de dispersión de relación entre número de camas y ratings
def beds_ratings_scatter(df):

    fig = px.scatter(df, x='Camas', y='Ratings', color='Property_types',
                    title="Relación entre Número de Camas y Ratings",
                    labels={"Camas": "Número de Camas", "Ratings": "Ratings"})
    st.plotly_chart(fig, use_container_width=True)













