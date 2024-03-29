import pickle
import os
import sklearn
from sklearn import linear_model

def checkDir():
	if 'models' in os.listdir('../'): 
		return True
	return False

def makeDir():
	if checkDir(): 
		pass
	else: 
		os.mkdir('../models')

# will save a model at ../models and will return the location+name of saved model
def saveModel(modelClass, name = None):
	fileName = name
	if name is None: 
		fileName = 'model'+str(len(os.listdir('../models')))
	fileName+='.sav'
	pickle.dump(modelClass, open('../models/'+fileName, 'wb'))
	return '../models/'+fileName

# model will be loaded through the location of model that is returned from the 
def loadModel(fileName):
	model1 = pickle.load(open(fileName, 'rb'))
	return model1

def main():
    # Crear un botón para ejecutar el modelo
    if st.button('Generar texto'):
        # Crear un hilo y ejecutar el modelo en ese hilo
        t = threading.Thread(target=run_model)
        t.start()
### All the below tests passed
if __name__ == '__main__':
	print(checkDir())
	makeDir()
	reg = linear_model.Ridge(alpha = 0.5)
	reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
	print("og Coeff: ",reg.coef_)
	path = saveModel(reg)
	print("Model Name: "+path)
	model = loadModel(path)
	print("Loaded Model:", model1.coef_)
