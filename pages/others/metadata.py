# Load important libraries 
import pandas as pd
import streamlit as st 
import os
import warnings
warnings.filterwarnings('ignore')

def app():
    # Load the uploaded data 
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Cargue los datos a través de la página *Cargar datos*")
    else:
        data = pd.read_csv('data/main_data.csv')
        st.dataframe(data)

        # Read the column meta data for this dataset 
        col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')

}
        '''
        st.markdown("#### Cambia la informacion del tipo de columna")
        
        # Use two column technique 
        col1, col2 = st.beta_columns(2)

        global name, type
        # Design column 1 
        name = col1.selectbox("Seleccion la columna", data.columns)
        
        # Design column two 
        current_type = col_metadata[col_metadata['column_name'] == name]['type'].values[0]
        print(current_type)
        column_options = ['numerical', 'categorical','object']
        current_index = column_options.index(current_type)
        
        type = col2.selectbox("Seleciona el tipo de columna", options=column_options, index = current_index)
        
        st.write("""Seleccione el nombre de su columna y el nuevo tipo de los datos.
                    Para enviar todos los cambios, haga clic en *Enviar cambios* """)

        
        if st.button("Cambia el tipo de columna"): 

            # Set the value in the metadata and resave the file 
            # col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')
            st.dataframe(col_metadata[col_metadata['column_name'] == name])
            
            col_metadata.loc[col_metadata['column_name'] == name, 'type'] = type
            col_metadata.to_csv('data/metadata/column_type_desc.csv', index = False)

            st.write("Tus cambios se han realizado!")
            st.dataframe(col_metadata[col_metadata['column_name'] == name])
