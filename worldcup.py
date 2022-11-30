import streamlit as st
import streamlit_book as stb
from pathlib import Path

def main():
    # Streamlit webpage properties
    st.set_page_config(layout="wide", page_title="WorldCup Demo 0.0.1",
                   page_icon="🦖",
                   initial_sidebar_state="expanded")

    # Streamit book properties
    current_path = Path(__file__).parent.absolute()
    
    stb.set_book_config(paths="pages",
                        options=[
                                      "Resultados das Copas do Mundo",   
                                      "Resultados das Partidas", 
                                      "Jogadores", 
                                      "Clube de Origem", 
                                      ])


if __name__ == "__main__":
       main()
