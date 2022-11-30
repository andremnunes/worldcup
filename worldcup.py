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
    
    stb.set_book_config(menu_title="Datasaurus Rex",
                    menu_icon="info-square",
                    options=[
                            "Welcome!",
                            "What is a Datasaurus?",
                            "Where can I see one?",
                            "Can I create a Datasaurus?",
                            "About"
                            ],
                    paths=[
                            "docs/01_resultado_copa_mundo",
                            "docs/01_intro.py",
                            "docs/03_datasaurus.py",
                            "docs/04_custom.py",
                            "docs/05_about.py"
                          ],


if __name__ == "__main__":
       main()
