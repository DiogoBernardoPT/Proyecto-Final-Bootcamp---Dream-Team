import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Función para el gráfico de distribución de valoraciones
def rating_distribution(df):
    st.subheader('Distribución de Ratings')
    rating_hist = px.histogram(df, x='ratings', nbins=20, title='Distribución de Ratings')
    st.plotly_chart(rating_hist, use_container_width=True, key='rating_hist')


# Función para el gráfico de precio por tipo de propiedad
def price_property_types(df):
    # st.header("Precio por Tipo de Propiedad")
    price_chart = px.box(df, x='property_types', y='prices_per_night', title='Precios por Tipo de Alojamiento')
    st.plotly_chart(price_chart, use_container_width=True, key='price_chart')

# Función para la gráfica de visualización entre Precio/Ratings por tipo de alojamiento
def price_rating_distribution(df):
    price_rating_distr_log = px.scatter(
        data_frame=df,
        x='ratings',
        y='prices_per_night',
        log_x=True,
        color='property_types',
        title='Distribución de Precio y Valoración por Tipo de Alojamiento (Escala Logarítmica)',
        labels={'ratings': 'Valoración (escala logarítmica)', 'prices_per_night': 'Precio por Noche'},
    )
    st.plotly_chart(price_rating_distr_log, use_container_width=True)

# Función para la gráfica de precio medio según la capacidad máxima de huéspedes
def average_price_by_capacity(df):
    avg_price = df.groupby('maximum_guests')['prices_per_night'].mean().reset_index()
    fig = px.bar(avg_price, x='maximum_guests', y='prices_per_night', 
                title='Precio Medio por Capacidad Máxima de Huéspedes')
    st.plotly_chart(fig, use_container_width=True)       

# Función para la gráfica de disperción de precio
def price_distribution_histogram(df):
    fig = px.histogram(df, x='prices_per_night', nbins=30, 
                    title='Distribución de Precios por Noche')
    st.plotly_chart(fig, use_container_width=True)

# Función para la gráfica de distribución de precios en intervalos
def price_distribution_bar_chart(df):
    # Agrupar por intervalos de precio
    intervalos = pd.cut(df['prices_per_night'], bins=10)
    conteo_por_intervalo = df.groupby(intervalos).size()

    bar_chart = px.bar(x=conteo_por_intervalo.index.astype(str), 
                        y=conteo_por_intervalo.values, 
                        labels={'x': 'Intervalos de Precio', 'y': 'Conteo'},
                        title='Distribución de Precios en el Rango Seleccionado')
    
    st.plotly_chart(bar_chart, use_container_width=True)

# Función para la gráfica de dispersión de precios en relación al número de reseñas
def reviews_price_scatter(df):
    price_hist = px.scatter(df, x='num_reviews', y='prices_per_night', 
                    title='Relación entre Número de Reseñas y Precio por Noche')
    st.plotly_chart(price_hist, use_container_width=True)    
# Mapa de Correlación
def correlation(df):

    df_corr = df.select_dtypes(include=['float64', 'int64'])
    
    correlation_matrix = df_corr.corr()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Mapa de Correlación')
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

# Función para la gráfica de distribución de Reseñas y Valoraciones por tipo de alojamiento
def reviews_rating_distribution(df):
    reviews_rating_distr = px.scatter(
        data_frame=df,
        x='ratings',
        y='num_reviews',
        log_x=True,
        color='prices_per_night',
        title='Distribución de Reseñas y Valoración por Tipo de Alojamiento (Escala Logarítmica)',
        hover_name='property_types',
        opacity=0.5,
        size='prices_per_night',
        size_max=15,
    )
    st.plotly_chart(reviews_rating_distr, use_container_width=True)

# Función para la gráfica de distribución de tipos de propiedades
def property_type_distribution(df):
    property_count = df['property_types'].value_counts()
    fig = px.pie(property_count, values=property_count.values, names=property_count.index, 
                title='Distribución de Tipos de Propiedades')
    st.plotly_chart(fig, use_container_width=True)

# Función para la gráfica de histograma de número de camas
def beds_distribution_histogram(df):
    fig = px.histogram(df, x='camas', nbins=30, 
                    title='Distribución del Número de Camas')
    st.plotly_chart(fig, use_container_width=True)

# Función para la gráfica scatter plot de relación entre número de camas y las valoraciones
def beds_ratings_scatter(df):
    fig = px.scatter(df, x='camas', y='ratings', color='property_types',
                    title='Relación entre Número de Camas y Ratings',
                    labels={'camas': 'Número de Camas', 'ratings': 'Ratings'})
    st.plotly_chart(fig, use_container_width=True)
















