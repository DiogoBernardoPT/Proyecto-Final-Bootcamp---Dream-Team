import streamlit as st
import pandas as pd
from modules import visualizations


def show(df):
    st.title('**Discover the world of Airbnb accommodations through data!**')
    st.write(''' 
    This dashboard allows you to explore prices, ratings, and property locations in the city of Barcelona. 
    Analyze trends, filter based on your preferences, and uncover the most interesting insights 
    in every corner of the Airbnb market.  
    Ready to start exploring? 🏠✨
    ''')
    st.image('images/captivating_barcelona.png', width=900)

    st.header('Exploratory Data Analysis')
    st.write('''
    In this section, we will explore key insights from the Airbnb dataset. 
    We will analyze trends in pricing, property types, ratings, and reviews 
    to uncover patterns and correlations that can help better understand 
    the dynamics of the Airbnb market in Barcelona.
    ''')

    st.subheader('First Rows of the DataFrame')
    df_display = df.drop(columns=['urls', 'timestamp', 'record_id', 'titles', 'location', 'host_name'])
    st.dataframe(df_display.head())

    df_filtered = df.copy()

    # Filtros interactivos
    property_type = st.multiselect('Select property type:', options=df['property_types'].unique(), default=None)
    rating_options = ['1-2', '3-4', '4-5']
    selected_rating_range = st.selectbox('Select a rating range:', options=rating_options)
    num_reviews_slider = st.slider(
        'Select minimum number of reviews:',
        min_value=0,
        max_value=int(df['num_reviews'].max()),
        value=0,
        step=1,
        help="Move the slider to select the desired number of reviews."
    )
    price_range = st.slider('Price range per night:', 
                            min_value=int(df['prices_per_night'].min()), 
                            max_value=int(df['prices_per_night'].max()), 
                            value=(int(df['prices_per_night'].min()), int(df['prices_per_night'].max())), 
                            format='€%d')

    apply_filters = st.button('Apply Filters')

    if apply_filters:
        if selected_rating_range == '1-2':
            df_filtered = df_filtered[(df_filtered['ratings'] >= 1) & (df_filtered['ratings'] <= 2)]
        elif selected_rating_range == '3-4':
            df_filtered = df_filtered[(df_filtered['ratings'] > 2) & (df_filtered['ratings'] <= 4)]
        elif selected_rating_range == '4-5':
            df_filtered = df_filtered[(df_filtered['ratings'] > 4) & (df_filtered['ratings'] <= 5)]
        
        if num_reviews_slider > 0:
            df_filtered = df_filtered[df_filtered['num_reviews'] >= num_reviews_slider]
        
        if property_type:
            df_filtered = df_filtered[df_filtered['property_types'].isin(property_type)]
        
        # Aplicar el filtro del precio
        df_filtered = df_filtered[(df_filtered['prices_per_night'] >= price_range[0]) & 
                                  (df_filtered['prices_per_night'] <= price_range[1])]

        st.subheader('Filtered Listings')
        df_filtered_display = df_filtered.drop(columns=['urls', 'timestamp', 'record_id', 'titles', 'location', 'host_name'])
        st.dataframe(df_filtered_display.head())
    else:
        st.subheader('Listings Without Filters')
        df_display = df.drop(columns=['urls', 'timestamp', 'record_id', 'titles', 'location', 'host_name'])
        st.dataframe(df_display.head())

    # Visualizaciones EDA lado a lado
    st.subheader('**Data Visualizations**')
    col1, col2 = st.columns(2)
    
    with col1:
        visualizations.price_rating_distribution(df_filtered)
    with col2:
        visualizations.average_price_by_capacity(df_filtered)

    # Anadir hipotesis aqui
    with st.expander("Click to see insights from the graphs above"):
        st.write("""
        - **Property Type Distribution**: Breakdown of the different types of properties listed.
        - **Average Price by Maximum Guest Capacity**: Average price per night based on the maximum number of guests a property can accommodate.         
        """)
    
    # Visualizacion con relacion al precio
    st.header('Price Analysis')

    with st.expander("Correlation Map"):
        visualizations.correlation(df_filtered)

    col1, col2 = st.columns(2)
    with col1:
        visualizations.price_property_types(df_filtered)
    with col2:    
        visualizations.price_distribution_histogram(df_filtered)

    
    with st.expander("Click to see insights from the price analysis"):
        st.write("""
        - **Correlation**: A map showing how various factors correlate with the price per night.
        - **Price by Property Type**: A breakdown of how price varies by different property types.                  
        - **Price Distribution**: A histogram showing the distribution of prices across all properties.
        """)

    # Visualizaciones relacionadas a Reviews/Ratings
    col1, col2 = st.columns(2)
    with col1:
        visualizations.rating_distribution(df_filtered)
    with col2:
        visualizations.reviews_rating_distribution(df_filtered)

    visualizations.reviews_price_scatter(df_filtered)

    with st.expander("Click to see insights from the reviews and ratings visualizations"):
        st.write("""
        - **Rating Distribution**: Shows the distribution of properties based on their ratings.
        - **Reviews vs Rating**: Analyzes how the number of reviews correlates with property ratings.
        - **Reviews vs Price**: Shows how the number of reviews impacts the price per night.
        """)

    st.header('Top 10 Services Offered on Airbnb')
    df_servicios = pd.read_csv('data/df_servicios_final_cleaned.csv')
    visualizations.top_10_services_chart(df, df_servicios)

    with st.expander("Click to see insights from the services analysis"):
        st.write("""
    - **Top 10 Services**: Visualizes the top 10 most offered services on Airbnb in Barcelona.
    """)        