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

#st.set_page_config(page_title="GEO-AI Formulario",page_icon="🧊")

#portalUrl = "https://soluciones.aeroterra.com/portal"
#username = 'jguerra@ASA'
#password = "Mayo2021*"


""" def generateToken(username, password, portalUrl):
    # Retrieves a token to be used with API requests.
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    parameters = {'username': username,
                  'password': password,
                  'client': 'referer',
                  'referer': portalUrl,
                  'expiration': 60,
                  'f': 'json'}
    url = portalUrl + '/sharing/rest/generateToken?'
    response = requests.post(url, data=parameters, headers=headers)

    try:
        jsonResponse = response.json()
        
        if 'token' in jsonResponse:
            return jsonResponse['token']
        elif 'error' in jsonResponse:
            print (jsonResponse['error']['message'])
            for detail in jsonResponse['error']['details']:
                print (detail)
    except ValueError:
        print('An unspecified error occurred.')
        print(ValueError)

        
token = generateToken(username, password, portalUrl)

#gis = GIS("https://soluciones.aeroterra.com/portal", "jguerra@ASA", "Mayo2021*",verify_cert=False) """


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
	def __init__(self, data,filename='myfile',file_ext='csv'):
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


#gis = GIS("https://soluciones.aeroterra.com/portal/home", "jguerra@ASA", "Mayo2021*",verify_cert=False)
pd.options.mode.chained_assignment = None  # default='warn'
now = datetime.now()
def app():
        if  'main_data.csv' not in os.listdir('data'):
            st.markdown("Carga previamente el csv en el panel inicial!")
        else:
            data = pd.read_csv('data/main_data.csv')
            st.markdown("### Formulario de datos")
            def load_model():
                    modelo = pickle.load(open('data/metadata/regresion_lineal.pickle', 'rb'))
                    return modelo
            modelo = load_model()
            Material = st.selectbox('Valor 1',options=[   
                                'Plomo','Polietileno','Cobre','Hierro fundido','Hierro ductil','Cloruro de polivinilo','Hierro Galvanizado/Polietileno de alta densidad','Alta densidad plomo']) #'Cast Iron','Polyvinyl Chloride','Galvanized IronHigh-Density Polyethylene','High Probability Lead','Ductile Iron'])#,'Polyvinyl Chloride','Galvanized IronHigh-Density Polyethylene','High Probability Lead','Ductile Iron','Polyethylene','Copper'
            Material_Lead, Material_Polyethylene,Material_Copper,Material_Ductile_Iron,Material_Polyvinyl_Chloride,Material_Cast_Iron,Material_High_Probability_Lead,Material_Galvanized_IronHigh_Density_Polyethylene=0,0,0,0,0,0,0,0 #Material_Galvanized IronHigh-Density Polyethylene,Material_Polyvinyl Chloride,Material_High Probability Lead,Material_Ductile Iron, = 0,0,0,0,0,0,0,0 #,Material_Cast_Iron,Material_Galvanized_IronHigh_Density_Polyethylene,Material_Polyvinyl_Chloride,Material_High_Probability_Lead,Material_Ductile_Iron,Material_Polyethylene, Material_Copper
            if (Material == 'Plomo'):
                Material_Lead = 1
            elif (Material == 'Polietileno'):
                Material_Polyethylene = 1
            elif (Material == 'Cobre'):
                Material_Copper = 1
            elif (Material == 'Hierro fundido'):
                Material_Cast_Iron = 1
            elif (Material == 'Cloruro de polivinilo'):
                Material_Polyvinyl_Chloride = 1
            elif (Material == 'Hierro Galvanizado/Polietileno de alta densidad'):
                Material_Galvanized_IronHigh_Density_Polyethylene = 1
            elif (Material == 'Alta densidad plomo'):
                Material_High_Probability_Lead = 1
            elif (Material == 'Hierro ductil'):
                Material_Ductile_Iron = 1
            Longitud_millas = st.number_input('Valor 2',value=1.) #Longitud en millas
            Diametro_pulgadas = st.number_input('Valor 3',value=1.) # Diametro en pulgadas
            Fecha_instalacion = st.date_input(' Valor 4') #Fecha de instalación
            #st.write('Los valores ingresados son los siguientes {}'.format(
                    #[Material,Longitud_millas,
                    #Diametro_pulgadas,Fecha_instalacion]))
            def predict_aguas(Material_Lead,Material_Polyethylene,Material_Copper,Material_Ductile_Iron,Material_Polyvinyl_Chloride,Material_Cast_Iron,Material_High_Probability_Lead,Material_Galvanized_IronHigh_Density_Polyethylene,Longitud_millas,
                    Diametro_pulgadas):
                    input=np.array([[Material_Lead,Material_Polyethylene,Material_Copper,Material_Ductile_Iron,Material_Polyvinyl_Chloride,Material_Cast_Iron,Material_High_Probability_Lead,Material_Galvanized_IronHigh_Density_Polyethylene,Longitud_millas,
                    Diametro_pulgadas]]).astype(np.float64)
                    prediction = modelo.predict(input)
                    return round(float(prediction),1)
            if st.checkbox("Predicción", False):
                    output = predict_aguas(Material_Lead,Material_Polyethylene,Material_Copper,Material_Ductile_Iron,Material_Polyvinyl_Chloride,Material_Cast_Iron,Material_High_Probability_Lead,Material_Galvanized_IronHigh_Density_Polyethylene,Longitud_millas,Diametro_pulgadas) 
                    st.markdown("Los valores siguientes estan expresados en **años**")
                    st.success('La predicción para el tiempo de vida del segmento de red fue: ----------->  {}  ' .format(output))

                    if st.checkbox("Tabular y georreferenciar los datos", False):
                        st.subheader('Datos del formulario')
                        input=pd.DataFrame({'Material': [Material],
                        'longitud_millas': [Longitud_millas],
                        "diametro_pulgadas":[Diametro_pulgadas],
                        "fecha_instalacion":[Fecha_instalacion],
                        "tiempo_vida": [output]})
                        st.markdown("**1.- Valores ingresados y predichos**")
                        st.write("", input)
                        if st.checkbox("Desea agregar y georreferenciar los datos?", False):
                            st.subheader("Tablero de control")
                            if st.checkbox("Desea insertar los datos a la visualización del tablero?", False):
                                pipe = pd.DataFrame(data, columns=["id"])
                                id = st.selectbox('Seleccione el ID de la red', pipe)
                                Ult_fecha_ruptura = st.date_input("Ingrese la última fecha de ruptura")
                                costo_usd_ruptura = st.number_input("Ingrese el costo en (usd) de ruptura", min_value=10, max_value=35000)
                                if (Ult_fecha_ruptura <= Fecha_instalacion):
                                    st.warning("Alerta: Los datos no pueden ser ingresados si son iguales o anteriores a la fecha de instalacion")
                                    st.stop()
                                else:
                                    input_final=pd.DataFrame({"id":[id],
                                    "fecha_instalacion":[Fecha_instalacion],
                                    'material': [Material],
                                    'longitud_millas': [Longitud_millas],
                                    "diametro_pulgadas":[Diametro_pulgadas],
                                    "tiempo_vida": [output],
                                    "ult_fecha_ruptura":[Ult_fecha_ruptura],
                                    "costo_usd_ruptura":[costo_usd_ruptura]})
                                    input_final["fecha_instalacion"] = pd.to_datetime(input_final["fecha_instalacion"])
                                    input_final["fecha_instalacion"] = input_final["fecha_instalacion"].astype('int64')//1e9
                                    input_final["ult_fecha_ruptura"] = pd.to_datetime(input_final["ult_fecha_ruptura"])
                                    input_final["ult_fecha_ruptura"] = input_final["ult_fecha_ruptura"].astype('int64')//1e9
                                    st.markdown("**2.- Valores agregados por el usuario:**")
                                    #st.write("", input_final)
                                    prox_rup = pd.DataFrame({'proxima_ruptura':[0]})

                                    prox_rup["proxima_ruptura"] = input_final['ult_fecha_ruptura'] + input_final['tiempo_vida']*60*60*24*365
                                    prox_rup["proxima_ruptura"] = pd.to_datetime(prox_rup["proxima_ruptura"],unit='s')
                                    prox_rup["proxima_ruptura"] = prox_rup["proxima_ruptura"].astype(str).str[:10]
                                    prox_rup["proxima_ruptura"]=pd.to_datetime(prox_rup["proxima_ruptura"])

                                    prox_rup["tiempo_faltante"] = (prox_rup.proxima_ruptura - pd.Timestamp('now')).astype('timedelta64[D]').astype('int')/365.5 
                                    prox_rup["tiempo_faltante"] = (prox_rup["tiempo_faltante"]).astype(str).str[:4] 
                                    prox_rup["proxima_ruptura"] = prox_rup["proxima_ruptura"].astype(str).str[:10]
                                    
                                    st.markdown("**3.- Fecha de la próxima ruptura y tiempo restante: **")
                                    st.write("", prox_rup)
                                    
                                    difference = round(float(prox_rup["tiempo_faltante"]),2)
                                    inf_prob_prox_ruptura = (input_final.costo_usd_ruptura * difference)*0.10
                                    inf_prob_prox_ruptura = round(int(inf_prob_prox_ruptura),0)

                                    costo_prob_ruptura = pd.DataFrame({'costo_fut_usd_ruptura':[0]})
                                    costo_prob_ruptura['costo_fut_usd_ruptura'] = input_final.costo_usd_ruptura + inf_prob_prox_ruptura
                                    
                                    st.markdown("**4.- Costo de la proxima reparación, incluyendo sobrecargos (0,10): **")
                                    st.write("", costo_prob_ruptura)

                                    st.markdown("**5.- Valores finales a insertar: **")
                                    tabla_final =  pd.concat([input_final, prox_rup, costo_prob_ruptura], axis=1) 
                                    st.write("", tabla_final)
                                    
                                    if st.button("Descargar"):
                                        #st.dataframe(tabla_final)
                                        download = FileDownloader(tabla_final.to_csv(),file_ext='csv').download()
                                    
                                    #target_item = gis.content.get("a48f40f410b0426bb8ba93c521f3477b")
                                    #flayer = target_item.layers[0]
                                    
                                    st.markdown("**6.- Inserción de datos en el tablero **")
                                    if st.button("Insertar"):

                                        tabla_final[["ult_fecha_ruptura","fecha_instalacion"]] = tabla_final[["ult_fecha_ruptura","fecha_instalacion"]].astype(float).astype(int)
                                        tabla_final[["costo_fut_usd_ruptura","costo_usd_ruptura","tiempo_faltante"]] = tabla_final[["costo_fut_usd_ruptura","costo_usd_ruptura","tiempo_faltante"]].astype(str)
                                        #tabla_final[["fecha_instalacion","costo_fut_usd_ruptura","ult_fecha_ruptura","costo_usd_ruptura"]] = tabla_final[["fecha_instalacion","costo_fut_usd_ruptura","ult_fecha_ruptura","costo_usd_ruptura"]].astype('int64')
                                        fl = flayer.query().sdf[["id","SHAPE__Length","SHAPE", "objectid"]]
                                        join = pd.merge(left=fl, right =tabla_final,on = 'id')
                                        updt_features = FeatureSet.from_dataframe(join).features
                                        flayer.edit_features(updates=updt_features)
                                        df = flayer.query().sdf
                                        st.write("insertados!")

                                    #components.iframe("https://soluciones.aeroterra.com/portal/apps/opsdashboard/index.html#/950aae7c695846f3a0927f216e687163",width=700, height=820)
                                    st.markdown("<p style='text-align: center;'><a href='https://soluciones.aeroterra.com/portal/apps/opsdashboard/index.html#/950aae7c695846f3a0927f216e687163'>Abrir tablero en ventana externa</a></p>", unsafe_allow_html=True) 
                                    
                                    #target_item = gis.content.get("a48f40f410b0426bb8ba93c521f3477b")#   #88615be5d0224a9b9bf3c33532033750
                                    #flayer = target_item.layers[0]
                                    #df = flayer.query().sdf        
                                    
                                    st.markdown("**7.- Accediendo a los datos tabulares de la Feature: -Red de distribución- **")
                                    st.write('',df[['id','fecha_instalacion',"material","longitud_millas","diametro_pulgadas","tiempo_vida","ult_fecha_ruptura","costo_usd_ruptura","costo_fut_usd_ruptura","tiempo_faltante"]])

#########################################################################Codigos de prueba###########################################################################

                                     #st.dataframe(tabla_final)

                                    #target_item = gis.content.get("88615be5d0224a9b9bf3c33532033750")
                                    #flayer = target_item.layers[0]
                                    #df = flayer.query().sdf

                                    #if st.button("agregar") is True:
                                    #ports_features = flayer.features
                                    #sfo_feature = [f for f in ports_features if f.attributes['id']==tabla_final['id'][0]
                                    #sfo_feature.attributes
                                    #overlap_rows = pd.merge(left = df.sdf, right = tabla_final, how='inner',
                                        #    on='id')
                                        #    overlap_rows
                                            #updt_features = FeatureSet.from_dataframe(tabla_final).features
                                            #df.edit_features(adds = updt_features)
                                            
                                            #target_layer = target_item.layers[0]
                                            #fset = tabla_final.query()
                                            #sfo_feature.attributes
                                            #target_layer.edit_features(updates=[sfo_edit])
                                    #st.button("actualizar")
                                    #st.button("borrar")               


                    #costo_prob_ruptura['costo_usd_ruptura'] = costo_prob_ruptura
                    #costo_prob_ruptura['costo_usd_porc_cambio'] = ((costo_prob_ruptura['costo_usd_ruptura']  - input_final['costo_usd_ruptura'])/input_final['costo_usd_ruptura'])*100/10   #  df.pct_change(axis=1)['col2']#[['costo_usd_ruptura','inf_prob_prox_ruptura']]


                    #from sklearn.model_selection import train_test_split
                    #from sklearn.preprocessing import LabelEncoder
                    #from sklearn.preprocessing import OneHotEncoder
                    #from sklearn.linear_model import LinearRegression, LogisticRegression
                    #from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier        
                            
                    #if (grabado == "Mostrar"):
                    # st.write("Grabar datos en Tabla")
                    #st.write('los valores seleccionados son los siguientes {}'.format(
                    #[Material,Longitud_millas,Diametro_pulgadas,Fecha_instalacion,output])) """
                        
                    #print("Testeo score: {0:.2f} %".format(100 * score))
                    #X_train, X_test, y_train, y_test = train_test_split(df, output, test_size=0.2, random_state=4)
                    #st.write("Test score: {0:.2f} %".format(100 * score))
                    #prediction = modelo.predict(input)
                    #score = modelo.score(input, output)
                    #st.write("Test score: {0:.2f} %".format(100 * score))
                    #Ypredict = pickle_model.predict(Xtest)
                    #print(Ypredict)
                    #a =input.reshape(-1,1)
                    #score = modelo.score(output, input)
                    #st.write('model.'.format(score))

                    #.strftime('%Y-%m-%d') 
                    #prox_rup["proxima_ruptura"] = pd.to_datetime(prox_rup["proxima_ruptura"]) #datetime.datetime.strptime(prox_rup["proxima_ruptura"], '%Y-%m-%d')
                    #st.write("Próxima reparación: ", prox_rup)
                    #prox_rup["proxima_ruptura"] = pd.to_datetime(prox_rup["proxima_ruptura"])


                     #costo_prob_ruptura

                    #prox_rup["proxima_ruptura"] = float(prox_rup["proxima_ruptura"].days)
                    #format = now.strftime('%Y-%m-%d')
                    #now2 = pd.DataFrame(format)
                    #st.write("now: ", now2)
                    #diferencia_dia_año = abs((prox_rup["proxima_ruptura"] - now).days/365.5)
                    #format = now.strftime('%Y-%m-%d')
                    #st.write("diffe: ", diferencia_dia_año)
                    #diferencia_dia_año = abs((prox_ruptura - now).days/365.5)
                    #diferencia_dia_año =   round(float(diferencia_dia_año),2)
                    #infla_prob_prox_ruptura = (prox_ruptura * diferencia_dia_año)*0.10
                    #infla_prob_prox_ruptura = round(int(infla_prob_prox_ruptura),0)
                    #costo_prob_ruptura= prox_ruptura + infla_prob_prox_ruptura """
                    #form.text_input(label='Enter some text')    #text_input = st.write(label='Enter some text')
                    #submit_button = form.form_submit_button(label='Submit') #st.form_submit_button(label='Agregar') 
                    #else:

                    #prox_rup["tiempo_faltante"] = round(float(prox_rup["tiempo_faltante"]),1)
                    #inf_prob_prox_ruptura = (prox_rup.proxima_ruptura * prox_rup.tiempo_faltante)*0.10

                    #difference = round(float(difference),2)
