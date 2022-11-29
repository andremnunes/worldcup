import streamlit as st
import streamlit_book as stb
from pathlib import Path

def main():
       # Streamlit webpage properties
       st.set_page_config(layout="wide", page_icon="📚", page_title="WorldCup Demo 0.0.1")

       # Streamit book properties
       save_answers = True
       current_path = Path(__file__).parent.absolute()
       stb.set_book_config(menu_title="streamlit_book",
                            menu_icon="lightbulb",
                            options=[
                                          "Resultados das Copas do Mundo",   
                                          "Resultados das Partidas", 
                                          "Jogadores", 
                                          "Clube de Origem", 
                                          "Convocações"
                                          ], 
                            paths=[
                                          current_path / "pages/00_capa.md", 
                                          current_path / "pages/01_resultado_copa_mundo.py",
                                          current_path / "pages/02_resultado_partidas.py",
                                          current_path / "pages/03_jogadores.py",
                                          current_path / "pages/04_clubes_origem.py",
                                   ],
                            icons=[
                                          "code", 
                                          "robot", 
                                          "book", 
                                          "pin-angle", 
                                          "shield-lock"
                                   ],
                            save_answers=save_answers,
                            )


if __name__ == "__main__":
       main()
