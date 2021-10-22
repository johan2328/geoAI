# Import necessary libraries
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import streamlit as st
import pickle
# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

#st.set_page_config(page_title="GEO-AI validacion",page_icon="🧊")

from .utils import isNumerical
import os

def app():
    """Esta aplicación ayuda a ejecutar modelos de aprendizaje automático sin tener que escribir código explícito
    por el usuario. Ejecuta algunos modelos básicos y permite que el usuario seleccione las variables X e y. 
    """
    
    # Load the data 
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Cargue los datos a través de la página: Cargar datos!")
        #data=data.fillna('')
    else:
        data = pd.read_csv('data/main_data.csv',verbose =True,keep_default_na=False,na_values=[''],warn_bad_lines = True, error_bad_lines=False)
        #data=data.fillna('')
        # Diccionario de parametros
        params = {}

        # Division en dos columnas
        st.empty()
        st.markdown('<br></br>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        # Diseño column 1
        y_var = col1.radio("Seleccione la variable a predecir (y)", options=data.columns)

        # Diseño column 2
        X_var = col2.multiselect("Seleccione las variables que se utilizarán para la predicción (X)", options=data.columns)

        # Check si el tamaño de X no es 0
        if len(X_var) == 0:
            st.error("Tienes que poner alguna variable X y no se puede dejar vacía.")
        #else:
            st.stop()
        # Check si y no esta X 
        if y_var in X_var:
            st.error("¡Advertencia! La variable Y no puede estar presente en su variable X.")

        # Opciones de prediccion
        pred_type = st.radio("Selecciona el tipo de modelo que quieres correr.", 
                            options=["Regresion", "Clasificacion"], 
                            help="Selecciona reg o clasificación")

        # Params del modelo
        params = {
                'X': X_var,
                'y': y_var, 
                'pred_tipo': pred_type,
        }

        st.write(f"**Variable a predecir:** {y_var}")
        st.write(f"**Variable que se utilizará para la predicción:** {X_var}")
        
        # Divide el dataset en train/test
        X = data[X_var]
        y = data[y_var]

        
        # Enconding
        X = pd.get_dummies(X)

        # Check si necesita encodear
        if not isNumerical(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            
        # Print de clasess totales
            st.write("Las clases y la clase que se les asigna es la siguiente:-")
            classes = list(le.classes_)
            for i in range(len(classes)):
                st.write(f"{classes[i]} --> {i}")
        

        # Slider de train / test split
        st.markdown("#### Train/test split")
        size = st.slider("Seleciona, deslizando la barra, el porcentaje de división de valores",
                            min_value=0.1, 
                            max_value=0.9, 
                            step = 0.1, 
                            value=0.8, 
                            help="Este es el valor que se utilizará para dividir los datos para entrenamiento y prueba. Por defecto = 80%")

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
        st.write("Número de muestras de entrenamiento:", X_train.shape[0])
        st.write("Número de muestras de entrenamiento test:", X_test.shape[0])

        # Save 
        with open('data/metadata/model_params.json', 'w') as json_file:
            json.dump(params, json_file)

        with open('data/metadata/model_params.pkl', 'wb') as pickle_file:
            pickle.dump(params, pickle_file)
        
        ''' RUNNING THE MACHINE LEARNING MODELS '''
        if pred_type == "Regresion":
            st.write("Running regresion.....")

            # Tabla del model y la precision(accurcy) 
            model_r2 = []

            # Linear regression 
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_r2 = lr_model.score(X_test, y_test)
            model_r2.append(['Regresion lineal', lr_r2])

            #rf_pickle = open('regresion_lineal.pickle', 'wb')
            #pickle.dump(lr_model, rf_pickle)
            #rf_pickle.close()

            # Decision Tree 
            dt_model = DecisionTreeRegressor()
            dt_model.fit(X_train, y_train)
            dt_r2 = dt_model.score(X_test, y_test)
            model_r2.append(['Alg. Arbol de decision - Regresion', dt_r2])

            # Salvar uno de los modelos 
            if dt_r2 > lr_r2:
                # save decision tree 
                #joblib.dump(dt_model, 'data/metadata/regresion_lineal_1.pickle')
                rf_pickle = open('data/metadata/regresion_lineal2.pickle', 'wb')
                pickle.dump(lr_model, rf_pickle)
                #rf_pickle.close()
            else: 
                #joblib.dump(lr_model, 'data/metadata/regresion_lineal_1.pickle')
                rf_pickle = open('data/metadata/regresion_lineal2.pickle', 'wb')
                pickle.dump(dt_model, rf_pickle)
                #rf_pickle.close()
            # Dataset de resultados
            results = pd.DataFrame(model_r2, columns=['Models', 'R2 Score']).sort_values(by='R2 Score', ascending=False)
            st.dataframe(results)
        
        if pred_type == "Clasificacion":
            st.write("Corriendo Classficación...")

            # Tabla del model y la precision(accurcy) 
            model_acc = []

            # Linear regression
            lc_model = LogisticRegression()
            lc_model.fit(X_train, y_train)
            lc_acc = lc_model.score(X_test, y_test)
            model_acc.append(['Regresion logistica', lc_acc])

            # Decision Tree
            dtc_model = DecisionTreeClassifier()
            dtc_model.fit(X_train, y_train)
            dtc_acc = dtc_model.score(X_test, y_test)
            model_acc.append(['Alg. Arbol de decision - clasificacion', dtc_acc])

            # Salvar uno de los modelos 
            if dtc_acc > lc_acc:
                # salvar decision tree 
                joblib.dump(dtc_model, 'data/metadata/model_classification.sav')
            else: 
                joblib.dump(lc_model, 'data/metadata/model_classificaton.sav')

            # Dataset de resultados 
            results = pd.DataFrame(model_acc, columns=['Models', 'Presición']).sort_values(by='Presición', ascending=False)
            st.dataframe(results)


