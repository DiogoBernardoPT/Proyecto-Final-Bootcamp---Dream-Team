import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Exploratory Data Analysis
# =====================================

# Función para el gráfico de distribución de valoraciones
def rating_distribution(df):
    st.subheader('Ratings Distribution')
    rating_hist = px.histogram(df, x='ratings', nbins=20)
    rating_hist.update_traces(marker_line_width=1, marker_line_color='black')
    st.plotly_chart(rating_hist, use_container_width=True, key='rating_hist')


# Función para la gráfica de dispersión de precio vs número de reseñas
def reviews_price_scatter(df):
    st.subheader('Number of Reviews vs Price per Night')
    price_hist = px.scatter(df, x='num_reviews', y='prices_per_night')
    st.plotly_chart(price_hist, use_container_width=True)

# Función para la gráfica de distribución de reseñas y valoraciones por tipo de alojamiento
def reviews_rating_distribution(df):
    st.subheader('Reviews and Ratings Distribution by Property Type')
    reviews_rating_distr = px.scatter(
        data_frame=df,
        x='ratings',
        y='num_reviews',
        log_x=True,
        color='prices_per_night',
        hover_name='property_types',
        opacity=0.5,
        size='prices_per_night',
        size_max=15,
    )
    st.plotly_chart(reviews_rating_distr, use_container_width=True)


# Prices Visualizations
# =====================================

# Mapa de correlación
def correlation(df):
    st.subheader('Correlation Map')
    df_corr = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = df_corr.corr()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    
    st.pyplot(fig)

# Función para el gráfico de precio por tipo de propiedad
def price_property_types(df):
    st.subheader('Prices by Property Type')
    price_chart = px.box(df, x='property_types', y='prices_per_night')
    st.plotly_chart(price_chart, use_container_width=True, key='price_chart')

# Función para la gráfica de visualización entre Precio/Valoraciones por tipo de alojamiento
def price_rating_distribution(df):
    st.subheader('Price and Rating Distribution by Property Type')
    price_rating_distr_log = px.scatter(
        data_frame=df,
        x='ratings',
        y='prices_per_night',
        log_x=True,
        color='property_types',
    )
    st.plotly_chart(price_rating_distr_log, use_container_width=True)

# Función para la gráfica de precio medio según la capacidad máxima de huéspedes
def average_price_by_capacity(df):
    st.subheader('Average Price by Maximum Guest Capacity')
    avg_price = df.groupby('maximum_guests')['prices_per_night'].mean().reset_index()
    fig = px.bar(avg_price, x='maximum_guests', y='prices_per_night')
    st.plotly_chart(fig, use_container_width=True)

# Función para la gráfica de dispersión de precios
def price_distribution_histogram(df):
    st.subheader('Price per Night Distribution')
    fig = px.histogram(df, x='prices_per_night', nbins=30)
    fig.update_traces(marker_line_width=1, marker_line_color='black')
    st.plotly_chart(fig, use_container_width=True)


# Servicios
# =====================================

def top_10_services_chart(df1, df2):
    # Hacemos merge del df de la habitacion con el de servicios
    df3 = pd.merge(left=df1, right=df2, how='inner', on="urls")

    df_services_merged = df3[['urls', 'prices_per_night', 'category', 'services']]

    # Averiguamos los 10 servicios mas ofrecidos
    top_10_services = df_services_merged.groupby("services").agg({"prices_per_night": "count"}).sort_values("prices_per_night", ascending=False).head(10).index
    top_10_df = df_services_merged[df_services_merged["services"].isin(top_10_services)]

    fig = px.bar(top_10_df, 
                 x="services", 
                 title="Top 10 Serviços Mais Oferecidos no Airbnb", 
                 labels={"services": "Serviços", "prices_per_night": "Conteúdo"},
                 category_orders={"services": top_10_services},
                 color="services", 
                 text_auto=True)

    st.plotly_chart(fig, use_container_width=True)    














