# Import necessary libraries
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import streamlit as st

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Custom classes 
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
        # Create the model parameters dictionary 
        params = {}

        # Use two column technique 
        col1, col2 = st.beta_columns(2)

        # Design column 1 
        y_var = col1.radio("Seleccione la variable a predecir (y)", options=data.columns)

        # Design column 2 
        X_var = col2.multiselect("Seleccione las variables que se utilizarán para la predicción (X)", options=data.columns)

        # Check if len of x is not zero 
        if len(X_var) == 0:
            st.error("Tienes que poner alguna variable X y no se puede dejar vacía.")
        #else:
            st.stop()
        # Check if y not in X 
        if y_var in X_var:
            st.error("¡Advertencia! La variable Y no puede estar presente en su variable X.")

        # Option to select predition type 
        # Option to select predition type 
        pred_type = st.radio("Selecciona el tipo de modelo que quieres correr.", 
                            options=["Regresión", "Clasificación"], 
                            help="Selecciona reg o clasificación")

        # Add to model parameters 
        params = {
                'X': X_var,
                'y': y_var, 
                'pred_tipo': pred_type,
        }

        # if st.button("Run Models"):

        st.write(f"**Variable a predecir:** {y_var}")
        st.write(f"**Variable que se utilizará para la predicción:** {X_var}")
        
        # Divide the data into test and train set 
        X = data[X_var]
        y = data[y_var]

        # Perform data imputation 
        # st.write("THIS IS WHERE DATA IMPUTATION WILL HAPPEN")
        
        # Perform encoding
        X = pd.get_dummies(X)

        # Check if y needs to be encoded
        if not isNumerical(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            # Print all the classes 
            st.write("Las clases y la clase que se les asigna es la siguiente:-")
            classes = list(le.classes_)
            for i in range(len(classes)):
                st.write(f"{classes[i]} --> {i}")
        

        # Perform train test splits 
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

        # Save the model params as a json file
        with open('data/metadata/model_params.json', 'w') as json_file:
            json.dump(params, json_file)

        ''' RUNNING THE MACHINE LEARNING MODELS '''
        if pred_type == "Regresión":
            st.write("Running regresión.....")

            # Table to store model and accurcy 
            model_r2 = []

            # Linear regression model 
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_r2 = lr_model.score(X_test, y_test)
            model_r2.append(['Regresión lineal', lr_r2])

            # Decision Tree model 
            dt_model = DecisionTreeRegressor()
            dt_model.fit(X_train, y_train)
            dt_r2 = dt_model.score(X_test, y_test)
            model_r2.append(['Alg. Arbol de decisión - Regresión', dt_r2])

            # Save one of the models 
            if dt_r2 > lr_r2:
                # save decision tree 
                joblib.dump(dt_model, 'data/metadata/model_reg.sav')
            else: 
                joblib.dump(lr_model, 'data/metadata/model_reg.sav')

            # Make a dataframe of results 
            results = pd.DataFrame(model_r2, columns=['Models', 'R2 Score']).sort_values(by='R2 Score', ascending=False)
            st.dataframe(results)
        
        if pred_type == "Clasificación":
            st.write("Running Classficacion...")

            # Table to store model and accurcy 
            model_acc = []

            # Linear regression model 
            lc_model = LogisticRegression()
            lc_model.fit(X_train, y_train)
            lc_acc = lc_model.score(X_test, y_test)
            model_acc.append(['Regresión lineal', lc_acc])

            # Decision Tree model 
            dtc_model = DecisionTreeClassifier()
            dtc_model.fit(X_train, y_train)
            dtc_acc = dtc_model.score(X_test, y_test)
            model_acc.append(['Alg. Arbol de decision - clasificacions', dtc_acc])

            # Save one of the models 
            if dtc_acc > lc_acc:
                # save decision tree 
                joblib.dump(dtc_model, 'data/metadata/model_classification.sav')
            else: 
                joblib.dump(lc_model, 'data/metadata/model_classificaton.sav')

            # Make a dataframe of results 
            results = pd.DataFrame(model_acc, columns=['Models', 'Presición']).sort_values(by='Presición', ascending=False)
            st.dataframe(results)



rf_pickle = open('data/metadata/model_params.pickle', 'rb')
#map_pickle = open('output_penguin.pickle', 'rb')
rfc = pickle.load(rf_pickle)
#unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
#map_pickle.close()
Material = st.selectbox('Material', options=['p1', 'p2'])
Fecha_instalación = st.number_input('Fecha_instalación', min_value=0)
Diametro pulgadas = st.number_input('Diametro en pulgadas', min_value=0)
Longitud_millas = st.number_input('Longitud en millas', min_value=0)
st.write('Has utilizado las siguientes variales {}'.format(
    [Material, sex, Fecha_instalación,
        Diametro pulgadas, Longitud_millas]))
