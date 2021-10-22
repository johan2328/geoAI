import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.figure import Figure
plt.style.use('dark_background')
def app():
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Cargue los datos a través de la página: Cargar datos")
    else:
        #df_analysis = pd.read_csv('data/2015.csv')
        df_analysis = pd.read_csv('data/main_data.csv')
        #df_visual = pd.DataFrame(df_analysis)
        df_visual = df_analysis.copy()
        cols = pd.read_csv('data/metadata/column_type_desc.csv')
        Categorical,Numerical,Object = utils.getColumnTypes(cols)
        cat_groups = {}
        unique_Category_val={}

        for i in range(len(Categorical)):
                unique_Category_val = {Categorical[i]: utils.mapunique(df_analysis, Categorical[i])}
                cat_groups = {Categorical[i]: df_visual.groupby(Categorical[i])}
                
        category = st.selectbox("Selección de categoría", Categorical + Object)

        sizes = (df_visual[category].value_counts()/df_visual[category].count()*4)

        labels = sizes.keys()

        maxIndex = np.argmax(np.array(sizes))
        explode = [0]*len(labels)
        explode[int(maxIndex)] = 0.1
        explode = tuple(explode)
        #plt.style.use('dark_background')
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes,explode = explode, labels=labels, autopct='%1.1f%%',shadow=False,startangle=0)
        ax1.axis('Equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title('Distribución para columna categórica - ' + (str)(category))
        st.pyplot(fig1)
        
        corr = df_analysis.corr(method='pearson')
        
        fig2, ax2 = plt.subplots()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = False
        # Colors
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0,ax=ax2)
        ax2.set_title("Matriz de correlación")
        st.pyplot(fig2)
        if Numerical == True:
                categoryObject=st.selectbox("Selección " + (str)(category),unique_Category_val[category])
                st.write(cat_groups[category].get_group(categoryObject).describe())
                colName = st.selectbox("Selección de columna ",Numerical)
                st.bar_chart(cat_groups[category].get_group(categoryObject)[colName])
        
        ## Code base to drop redundent columns """
 