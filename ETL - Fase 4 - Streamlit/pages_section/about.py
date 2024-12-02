import streamlit as st

def show():
    st.title("About Us")
    
    st.write("""
    Welcome to our project! This application was developed as part of our final project for the Data Science course. 
    Throughout this journey, we’ve applied our skills in data analysis and machine learning to create a powerful and interactive tool.
    We invite you to explore our work and connect with us below. We are always open to collaboration and new opportunities in the tech
    and data science field!

    Feel free to reach out to us through our profiles.
    """)

    # Criar duas colunas para os perfis de Diogo e Jesus
    col1, col2 = st.columns(2)

    # Perfil do Diogo
    with col1:
        st.subheader("Diogo")
        st.markdown("""
        **GitHub**: [GitHub Profile](https://github.com/DiogoBernardoPT)
        **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/diogogalhanas)
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg", width=50)
        


    # Perfil do Jesus
    with col2:
        st.subheader("Jesus")
        st.markdown("""
        **GitHub**: [GitHub Profile](https://github.com/jesus27mula)
        **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/jesus-maria-mulà-domènech)
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg", width=50)
