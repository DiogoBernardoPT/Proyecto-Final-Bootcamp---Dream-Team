import streamlit as st
import pandas as pd
from pages_section import landing_page, data_analysis, about


# Cargar el CSV en session_state
if 'df' not in st.session_state:
    file_path = 'data/df_final_cleaned.csv'
    df = pd.read_csv(file_path)
    # Almacenar el DataFrame limpio en session_state
    st.session_state.df = df

# Acceder al DataFrame directamente desde session_state
df = st.session_state.df

page = st.radio(
    "**🏠 Dear User, choose what you want to discover! 🏠**",
    ('Home', 'Data Analysis', 'About'),
    horizontal=True  # La barra de navegación aparece en horizontal
)

if page == 'Home':
    landing_page.show()
elif page == 'Data Analysis':
    data_analysis.show(df)
elif page == 'About':
    about.show()