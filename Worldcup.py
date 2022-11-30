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
                                      current_path  / "pages/1_Resultado_Copa_Mundo.py",
                                      current_path  / "pages/2_Resultado_Partidas.py",
                                      current_path  / "pages/3_Jogadores.py",
                                      current_path  / "pages/4_Clubes_Origem.py",
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
