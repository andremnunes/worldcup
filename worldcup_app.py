from PIL import Image
import streamlit as st
import streamlit_book as stb

# Streamlit webpage properties
im = Image.open("favicon.ico")
st.set_page_config(page_title="WorldCup", page_icon=im)

# Streamlit book properties
stb.set_book_config(path="pages")
