import os
import streamlit as st
import numpy as np
from PIL import  Image
import requests
import warnings
import streamlit.components.v1 as components
from multipage import MultiPage
import sys
warnings.filterwarnings('ignore')
st.set_page_config(page_title='Centro de entrenamiento GEO-AI',page_icon="🐙",initial_sidebar_state='auto')
from pages import carga_data, ml_validacion,visual_data,ml_formulario,ml_masivo# metadata # redundant # importadores de páginas 

hide_st = """
            <style>
               .viewerBadge_link__1S137{
                visibility: hidden;}
            </style>
          """
st.markdown(hide_st, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
# instancia de la app 
app = MultiPage()
#st.set_page_config (page_title = None, page_icon = None, layout = 'centered', initial_sidebar_state = 'auto')
# main del front

display = Image.open('Logo.png')
display = np.array(display)
display2 = Image.open('sub_logo.png')
display2 = np.array(display2)
col1,col2,col3=st.columns(3)
col1.image(display, width = 330)
col2=st.write('')
col3.image(display2, width = 240)

# aplicacion de la app
app.add_page("Selector de carga",  carga_data.app)
app.add_page("Análisis de Datos",visual_data.app)
app.add_page("Validación ML", ml_validacion.app)
app.add_page("Formulario ML", ml_formulario.app)
app.add_page("Predicción masiva ML", ml_masivo.app)
#app.add_page("Y-Parameter Optimization",redundant.app)

author_pic = Image.open('sidebar3.png')
st.sidebar.image(author_pic,use_column_width=True,width=70)

st.sidebar.write("")
# ;padding:3px
html_temp = """
    <div style="background:#040404">
    <h1 style="color:white;text-align:center; text-shadow:0em 0em 0.6em white"">Centro de entrenamiento GeoAI</h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html = True)

app.run()
st.markdown('<br></br>', unsafe_allow_html=True)
st.markdown('<br></br>', unsafe_allow_html=True)


st.sidebar.markdown("<p style='text-align: center;'><a href='https://www.aeroterra.com/es-ar/home'>Powered by Aeroterra</a></p>", unsafe_allow_html=True)


