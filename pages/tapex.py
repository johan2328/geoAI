import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
from PIL import  Image
import streamlit.components.v1 as components    
import warnings
import streamlit.components as stc
import base64
import transformers
from IPython.display import display 
from transformers import TapexTokenizer, BartForConditionalGeneration, AutoModelForCausalLM
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
from transformers.pipelines import pipeline
from transformers import pipeline
import torch
warnings.filterwarnings('ignore')                   

@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "tiiuae/falcon-40b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_text(prompt, max_length):
    tokenizer, model = load_model()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids=input_ids, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# import tempfile
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

#@st.cache
def app():
    st.markdown("", unsafe_allow_html=True)
    #st.markdown("<h2 style='text-align: center; color: #2e6c80;'>Predicción de rupturas en red de distribución de aguas</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #DBF2E9;'>Siga los pasos para entrenar en nuestros set de modelos de predictivos</h2>", unsafe_allow_html=True)
    #components.iframe("https://soluciones.aeroterra.com/portal/apps/webappviewer3d/index.html?id=3e5667a5b5634dfaace558c8e672976d",width=700, height=390) 
    global data
    #temp_file_path = data.to_csv('data/main_data.csv', index=False)
    #st.markdown("### Carga los CSV.") 
    st.write("\n")

    st.markdown("#### Cargar data (archivos .csv o .xlsx)")
     
    uploaded_file = st.file_uploader("Selecciona el archivo", type = ['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file,verbose =True,keep_default_na=False,na_values=[''],warn_bad_lines = True, error_bad_lines=False) #,verbose =True,keep_default_na=False,na_values=[''],warn_bad_lines = True, error_bad_lines=False
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)

    if st.button("Cargar"):
	    st.dataframe(data)
	    data.to_csv('data/main_data.csv', index=False)
	    numeric_cols = data.select_dtypes(include=['int64']).columns.tolist()
	    categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
	    
	    columns = []
	    try:
	            columns = utils.genMetaData(data)
	    except Exception as e:
		    print("Revisar sus valores en archivo: {}".format(e))
		    columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
		    columns_df.to_csv('data/metadata/column_type_desc.csv', index = False)
		    st.markdown("**Nombre de columna**-**Tipo de dato**")
		    for i in range(columns_df.shape[0]):
			    st.write(f"{i+1}. **{columns_df.iloc[i]['column_name']}** - {columns_df.iloc[i]['type']}")
			    st.markdown("Los anteriores son los tipos de columna automatizados detectados por la aplicación en los datos.") 
			    st.markdown("Pregunta a la tabla")


