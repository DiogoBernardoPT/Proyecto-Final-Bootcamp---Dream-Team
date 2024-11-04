import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from modules import visualizations

st.set_page_config(page_title='Dashboard de Airbnb', page_icon='🏠', layout='wide')

st.title('Dashboard de Análisis de Datos de Airbnb')
st.markdown(''' 
    **Bienvenido al punto de encuentro entre datos y destinos!**  
    Imagina descubrir los secretos de los alojamientos de Airbnb: desde precios y valoraciones hasta 
    ubicaciones sorprendentes. Sumérgete en este análisis interactivo y ajusta los filtros para explorar
    los datos a tu manera. El viaje comienza aquí!
''')

# Cargar los datos
uploader = st.file_uploader('Sube el archivo CSV de datos', type='csv')
if uploader:
    df = pd.read_csv(uploader)
    

# Barra lateral para navegar entre las pestañas
page = st.sidebar.selectbox('Navega por las secciones', ['Home', 'Data Analysis', 'Price Analysis', 'Locations', 'About'])

# Sección "Home"
if page == 'Home':
    st.title('Análisis de Datos de AirBnB')
    st.write(''' 
    **Descubre el mundo de los alojamientos en Airbnb a través de datos!**  
    Este dashboard te permitirá navegar por precios, calificaciones y ubicaciones de propiedades 
    en la ciudad. Explora tendencias, filtra por tus criterios preferidos, y encuentra las 
    conclusiones más interesantes en cada rincón del mercado de Airbnb.  
    Listo para comenzar la exploración? 🏠✨
    ''')
    st.image('images/airbnb_1stpage.png', use_column_width=True) # width=700)  # Ruta de la imagen en la carpeta 'images'

# Sección "Data Analysis"
elif page == 'Data Analysis':
    st.image('images/captivating_barcelona.png', use_column_width=True)
    st.header('Análisis Exploratorio de Datos')
    st.write('Gráficos y estadísticas de las propiedades de Airbnb.')

    # Mostrar DataFrame original
    st.subheader('Primeras Filas del DataFrame')
    st.dataframe(df.head())

    # Inicializar `df_filtered` como copia del DataFrame original
    df_filtered = df.copy()

    # Configurar controles de filtro
    property_type = st.multiselect('Selecciona el tipo de propiedad:', 
                                    options=df['Property_types'].unique(), 
                                    default=None)
    rating_options = ['1-2', '3-4', '4-5']
    selected_rating_range = st.selectbox('Selecciona un rango de Ratings:', options=rating_options)
    num_reviews_input = st.number_input('Ingresa el número mínimo de reseñas:', min_value=0, step=1)

    # Botón para aplicar filtros
    apply_filters = st.button('Aplicar Filtros')

    # Determinar el DataFrame a utilizar en las visualizaciones
    if apply_filters:
        # Aplicar filtros si el botón es pulsado
        if selected_rating_range == '1-2':
            df_filtered = df_filtered[(df_filtered['Ratings'] >= 1) & (df_filtered['Ratings'] <= 2)]
        elif selected_rating_range == '3-4':
            df_filtered = df_filtered[(df_filtered['Ratings'] > 2) & (df_filtered['Ratings'] <= 4)]
        elif selected_rating_range == '4-5':
            df_filtered = df_filtered[(df_filtered['Ratings'] > 4) & (df_filtered['Ratings'] <= 5)]
        
        if num_reviews_input > 0:
            df_filtered = df_filtered[df_filtered['Num_reviews'] >= num_reviews_input]
        
        if property_type:
            df_filtered = df_filtered[df_filtered['Property_types'].isin(property_type)]
        
        df_to_use = df_filtered  # Usar DataFrame filtrado en visualizaciones

        # Mostrar filas de alojamientos filtrados
        st.subheader('Alojamientos Filtrados')
        st.dataframe(df_filtered)
    else:
        # Sin filtros, usar el DataFrame original en visualizaciones
        df_to_use = df
        st.subheader('Alojamientos Sin Filtros Aplicados')
        st.dataframe(df)

    # Visualizaciones
    st.subheader('Visualizaciones')

    # Llamar a las funciones en visualizations.py
    visualizations.rating_distribution(df_to_use)
    visualizations.property_type_distribution(df_to_use)
    visualizations.beds_ratings_scatter(df_to_use)
    visualizations.beds_distribution_histogram(df_to_use)
    visualizations.reviews_rating_distribution(df_to_use)

    # Agregar otros gráficos relevantes para análisis de datos


# Sección "Price Analysis"
elif page == 'Price Analysis':
    st.header('Análisis de Precios')
    st.write(''' 
    En esta sección, exploraremos cómo los precios de las propiedades de Airbnb varían según el tipo de alojamiento y la ubicación. 
    Analizaremos las tendencias de precios, las distribuciones de ratings y la relación entre precios y número de reseñas. 
    Esto nos ayudará a comprender mejor el mercado de Airbnb y a identificar posibles patrones y oportunidades.
    ''')
    # Control deslizante para el rango de precios
    price_range = st.slider('Rango de precios por noche:', 
                            min_value=int(df['Prices_per_night'].min()), 
                            max_value=int(df['Prices_per_night'].max()), 
                            value=(int(df['Prices_per_night'].min()), int(df['Prices_per_night'].max())), 
                            format='€%d')  # Formato opcional para mostrar el símbolo del euro
    
    # Filtrar el DataFrame con el rango de precios
    df_filtered = df.copy()  # Asegúrate de que df_filtered esté basado en el DataFrame original
    df_filtered = df_filtered[(df_filtered['Prices_per_night'] >= price_range[0]) & 
                            (df_filtered['Prices_per_night'] <= price_range[1])]

    # Llamar las funciones en visualizations.py
    visualizations.price_property_types(df_filtered)
    visualizations.price_rating_distribution(df_filtered)
    visualizations.price_distribution_histogram(df_filtered)
    visualizations.average_price_by_capacity(df_filtered)
    visualizations.price_distribution_bar_chart(df_filtered)
    visualizations.reviews_price_scatter(df_filtered)

# Sección "Sobre"
elif page == 'About':
    st.header('Sobre el Proyecto')
    st.write(''' 
    Este dashboard fue creado como parte de un proyecto colaborativo de análisis de datos de Airbnb.
    Facilitadores : Diogo Bernardo, Jesús Mula, Sandra Mirambell         
    ''')

