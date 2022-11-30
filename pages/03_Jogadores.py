import streamlit as st
import streamlit_book as stb
from pathlib import Path

current_path = Path(__file__).parent.absolute()
stb.set_chapter_config(path=current_path  / '04_Clubes_Origem.py')
