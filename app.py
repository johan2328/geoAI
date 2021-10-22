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
#st.set_page_config(page_title="Ex-stream-ly Cool App",page_icon="🧊")
st.markdown('<br></br>', unsafe_allow_html=True)
st.markdown('<br></br>', unsafe_allow_html=True)


st.sidebar.markdown("<p style='text-align: center;'><a href='https://www.aeroterra.com/es-ar/home'>Powered by Aeroterra</a></p>", unsafe_allow_html=True)



#########################################################codigo de pruebas#################################################

#def load_lottieurl(url: str):
 #   r = requests.get(url)
 #   if r.status_code != 200:
 #      return None
 #  return r.json()



#author_pic = Image.open('sidebar.png')
#header_html = "<img src='C:\Users\jguerra\Documents\virtual_demo_aguas\streamlit_apps\clt_app\data-storyteller-main' class='img-fluid'>".format(img_to_bytes("sidebar.png"))
#st.sidebar.markdown(header_html, unsafe_allow_html=True)
#with open("style.css") as f:
#    st.sidebar.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#path = r"C:\Users\jguerra\Documents\virtual_demo_aguas\streamlit_apps\clt_app\data-storyteller-main\sidebar.png"
#image = Image.open(path)

#st.write("hello world")
#st.sidebar.image(image, width = 150)
#st.write("bye world")
#col1, col2, col3 = st.beta_columns([1,6,1])

#with col3:
 #   st.sidebar.write("2")
#st.sidebar.image(author_pic,use_column_width=True,width=)

#display4 = st_lottie(lottie_json,speed=0.5,width=200, height=300)
#display4 = np.array(display)
#col5, col6, col7  = st.beta_columns(3)
#col5.st.write('')
#col6.image(lottie_json,speed=0.5,width=200, height=300)
#col7.st.write('')

#st.markdown(st_lottie(lottie_json), unsafe_allow_html=True)

#def load_lottieurl(url: str):
#    r = requests.get(url)
#    if r.status_code != 200:
#        return None
#    return r.json()

#lottie_url = "https://assets4.lottiefiles.com/packages/lf20_4zbxcuwj.json"
#lottie_json = load_lottieurl(lottie_url)
#st_lottie(lottie_json,speed=0.5,width=200, height=300)

# Run del main app


#lottie_url = "https://assets4.lottiefiles.com/packages/lf20_4zbxcuwj.json"
#lottie_json = load_lottieurl(lottie_url)
#st_lottie(lottie_json,speed=0.5,width=200, height=300)

#st.title("Abalone Age Prediction")
