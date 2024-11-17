import streamlit as st

st.set_page_config(page_title='Airbnb Data Dashboard', page_icon='🏠', layout='wide')

def show():
    st.title('Welcome to Airbnb Data Analysis Dashboard')
    st.markdown('''
    **Welcome to the meeting point between data and destinations!**  
    Discover the secrets of Airbnb listings: from prices and ratings to surprising locations. 
    This dashboard allows you to explore prices, ratings, and property locations in the city of Barcelona. 
    Analyze trends, filter based on your preferences, and uncover the most interesting insights 
    in every corner of the Airbnb market.  
    Ready to start exploring? 🏠✨
    ''')
    st.image('images/airbnb_1stpage.png', width=900)