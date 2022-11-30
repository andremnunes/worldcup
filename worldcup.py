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
    
    stb.set_book_config(menu_title="streamlit_book",
                        menu_icon="lightbulb",
                        options=[
                                      "Resultados das Copas do Mundo",   
                                      "Resultados das Partidas", 
                                      "Jogadores", 
                                      "Clube de Origem", 
                                      ], 
                        paths=[
                                      "pages/01_resultado_copa_mundo/01.py",
                                      "pages/02_resultado_partidas.py",
                                      "pages/03_jogadores.py",
                                      "pages/04_clubes_origem.py",
                               ],
                        icons=[
                                      "code", 
                                      "robot", 
                                      "book", 
                                      "pin-angle", 
                                      "shield-lock"
                               ],
                        orientation=None, styles=None, save_answers=False
                        )


if __name__ == "__main__":
       main()
