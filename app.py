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

#st.sidebar.empty(label=None)
st.set_page_config(
    page_title='Centro de entrenamiento AI-ALlianz',
    page_icon="🤖",
    initial_sidebar_state="auto",
    menu_items=None
)
#st.set_page_config(page_title='Centro de entrenamiento AI',initial_sidebar_state="auto", menu_items=None)#,page_icon="🐙",
from pages import carga_data, ml_validacion,visual_data,ml_formulario,ml_masivo # metadata # redundant # importadores de páginas 

# Remove the icon
st.markdown("""
<style>
.streamlit-large-icon {
  display: none;
}
</style>
""", unsafe_allow_html=True)

# Remove the name
st.markdown("""
<style>
.streamlit-application-name {
  display: none;
}
</style>
""", unsafe_allow_html=True)

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

hide_st = """
            <style>
            .viewerBadge_link__1S137 {
                        padding: 1rem 1.25rem;
                        --bg-opacity: 1;
                        background-color: #ff4b4b;
                        background-color: rgba(255,75,75,var(--bg-opacity));
                        z-index: 10;
                        border-top-left-radius: 5px;
                        display: none;
                        visibility: hidden;
                                           }
            </style>
          """
st.markdown(hide_st, unsafe_allow_html=True)

hide_st2 = """
            <style>
                #root .styles_streamlitAppContainer__2rBcU.styles_stateContainer__29Rp6.viewerBadge_link__1S137{
                visibility: hidden;
                display: none;
                height: 0%;
                position: fixed;
                }
                svg {
                background-color: beige;
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                .viewerBadge_link__1S137 {
                        padding: 1rem 1.25rem;
                        --bg-opacity: 1;
                        background-color: #ff4b4b;
                        background-color: rgba(255,75,75,var(--bg-opacity));
                        z-index: 10;
                        border-top-left-radius: 5px;
                        display: none;
                        visibility: hidden;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
       """     
st.markdown(hide_st2, unsafe_allow_html=True)

hide_st_style = """
 <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# instancia de la app 
app = MultiPage()
#st.set_page_config (page_title = None, page_icon = None, layout = 'centered', initial_sidebar_state = 'auto')
# main del front
display = Image.open('grupo datco.png')
display = np.array(display)
display2 = Image.open('it4w.png')
display2 = np.array(display2)
col1,col2,col3=st.columns(3)
col1.image(display, width = 300)
col2=st.write('')
col3.image(display2, width = 200)

# aplicacion de la app
app.add_page("Selector de carga",  carga_data.app)
app.add_page("Análisis de Datos",visual_data.app)
app.add_page("Validación ML", ml_validacion.app)
app.add_page("Formulario ML", ml_formulario.app)
app.add_page("Predicción masiva ML", ml_masivo.app)
#app.add_page("Y-Parameter Optimization",redundant.app)

author_pic = Image.open('microsoft-partner.png')
st.sidebar.image(author_pic,use_column_width=True,width=70)

st.sidebar.write("")
# ;padding:3px

html_temp = """
    <div style="background:#040404">
    <h1 style="color:white;text-align:center; text-shadow:0em 0em 0.6em white"">Centro de entrenamiento AI</h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html = True)

app.run()
st.markdown('<br></br>', unsafe_allow_html=True)
st.markdown('<br></br>', unsafe_allow_html=True)


st.sidebar.markdown("<p style='text-align: center;'><a href='https://www.datco.net/'>Powered by Datco/it4w</a></p>", unsafe_allow_html=True)


