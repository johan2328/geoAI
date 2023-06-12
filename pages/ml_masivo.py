import streamlit as st
import numpy as np
import pandas as pd
import os
import datetime
from datetime import datetime
from datetime import timedelta
import warnings
import pickle
import streamlit.components as stc
import base64
from sklearn.metrics import r2_score
import streamlit.components.v1 as components
from IPython.display import display 
#from arcgis.gis import GIS
#from arcgis import GIS
#from arcgis import features
#from arcgis.features import FeatureSet, GeoAccessor
import time
import requests
warnings.filterwarnings('ignore')
timestr = time.strftime("%Y%m%d-%H%M%S")

def text_downloader(raw_text):
	b64 = base64.b64encode(raw_text.encode()).decode()
	new_filename = "new_text_file_{}_.txt".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)


def csv_downloader(data):
	csvfile = data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "new_text_file_{}_.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)

# Class
class FileDownloader(object):
	"""docstring for FileDownloader
	>>> download = FileDownloader(data,filename,file_ext).download()
	"""
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
		st.markdown("#### Descargar archivo ###")
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click aqui!!</a>'
		st.markdown(href,unsafe_allow_html=True)

def app():
    #def form_callback():
    #    st.write(st.session_state.my_button)
    """Esta aplicación ayuda a ejecutar modelos de aprendizaje automático sin tener que escribir código explícito
    por el usuario. Ejecuta algunos modelos básicos y permite que el usuario seleccione las variables X e y. 
    """
    st.write("\n")
    st.markdown("### -Predicción masiva-")
    st.write("\n")
    st.markdown("##### Cargar data (archivos .csv o .xlsx)")
    
    def load_model():
        modelo = pickle.load(open('data/metadata/regresion_lineal.pickle', 'rb'))
        return modelo
    modelo = load_model()

    st.write("\n")

    uploaded_file = st.file_uploader("Selecciona el archivo", type = ['csv', 'xlsx'])
    global data
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file,verbose =True,keep_default_na=False,na_values=[''],warn_bad_lines = True, error_bad_lines=False) #,verbose =True,keep_default_na=False,na_values=[''],warn_bad_lines = True, error_bad_lines=False
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)
    if st.checkbox("Cargar", False):
        data = data.drop(["Tiempo_vida","others"], axis = 1)
        st.write("Set de datos original", data)    

        dummy_ubicación = pd.get_dummies(data["ubicación"], prefix = "ubicación")
	dummy_genero = pd.get_dummies(data["Genero"], prefix = "genero")
	dummy_Nombre_auto = pd.get_dummies(data["Nombre_auto"], prefix = "Nombre_auto")
	dummy_Tipo_poliza = pd.get_dummies(data["Tipo_poliza"], prefix = "Tipo_poliza")
	
        data1 = data.drop(["Material","id","oid","Fecha_instalacion"], axis = 1)
        data2 = pd.concat([data1,dummy_ubicación,dummy_genero,dummy_Nombre_auto,dummy_Tipo_poliza], axis = 1)
        prediction_df = pd.DataFrame(data2)
        prediction = modelo.predict(prediction_df)
        predict_final = pd.DataFrame(prediction)
        predict_final.rename(columns ={0: "Predict_tiempo_vida"}, inplace = True)
        if st.checkbox("Predecir", False):
            with st.spinner('Espere por favor...'):
                time.sleep(3)
            st.success('Finaliza la predicción!')
            st.write("Valores de predicción", predict_final)
            if st.checkbox("Visualizar tabla final", False):
                salida =pd.concat([data,predict_final], axis=1,)
                st.write("Predicción final", salida)
                if st.button("Descargar"):
                    #st.dataframe(tabla_final)
                    download = FileDownloader(salida.to_csv(),file_ext='csv').download()
        #st.write("Definición de datos para el modelo* ----Se excluye por el momento Fecha instalación----", data2)
       #if st.button("Predecir", True):
        #if st.button("Predecir"):
            #prediction_df = pd.DataFrame(data2)
            #prediction = modelo.predict(prediction_df)
            #predict_final = pd.DataFrame(prediction)
            #predict_final.rename(columns ={0: "Predict_tiempo_vida"}, inplace = True)
            #st.write("Valores de predicción", predict_final)
            #predict_final = pd.DataFrame(prediction)

            #salida =pd.concat([data,predict_final], axis=1,)
            #st.write("predición final", salida)

