import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

################### uso de MLPClasiffier #########################################

print("################### MLPClasiffier #########################")
digits = load_digits() #cargamos los datos del conjunto digits

X= digits.data
Y= digits.target

#miramos las dimensiones de las entradas y salidas
print(f"X_shape= {X.shape}")
print(f"Y_shape= {Y.shape}")

#hacemos una particion de los datos con train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

#Hidden_layer_sizes indica la cantidad de capas ocultas y la cantidad de neuronas en cada una
#con hidden_layer_size= (5,6) quiere decir que queremos dos capas ocultas con 5 neuronas en la primera y 6 en la segunda
#no podemos controlar las capas de entrada y salida ya que las calcula usando el formato de los patrones que le pasemos
#activation permite elegir la funcion de activacion a usar
#activation= 'logistic' se usa la funcion sigmoidea entre 0 y 1
#solver ='sgd' indica que va a usar el metodo del gradiente
#learning_rate por defecto es constante, no cambia el coeficiente de aprendisaje dinamicamente
#learning_rate_init=0.01 indica que va usar un coeficiente de aprendisaje de 0.01 inicialmente
#max_iter=200 indica la cantidad maxima de epocas
#random_state permite reproducir la incializacion aleatoria de los pesos
#para que no de siempre lo mismo hay que sacarlo o ponerlo en NONE
#tol es la tolerancia de corte
#verbose=True indica que de salidas por consola de los resultados de la validacion, por defecto es false

eta=0.01
epocas=200
modelo = MLPClassifier(hidden_layer_sizes=(100,80),activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                       random_state=42,tol=1e-4,verbose=True)

modelo.fit(X_train,Y_train)#entrenamos el modelo

ratio_aciertos = modelo.score(X_test,Y_test) #ratio_aciertos = predicciones_correctas/N° de predicciones

print(f"Ratio de aciertos = {ratio_aciertos}")

# Fold con 5 particiones
print("####################### Utilizando KFold con 5 particiones ###################################")
eta =0.01
epocas=500
tol=1e-4

fk= KFold(n_splits=5)#lo dejamos con shufle en false

vector_aciertos=np.array([])
for i,(index_train,index_test) in enumerate(fk.split(X)):
    print(f"Fold= {i}")

    X_train = X[index_train]
    Y_train = Y[index_train]

    X_test = X[index_test]
    Y_test = Y[index_test]

    #definimos el modelo dentro del for para entrenar el mismo modelo con cada fold
    modelo = MLPClassifier(hidden_layer_sizes=(100,80),activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                       random_state=42, tol=tol, verbose=False) #para que no muestre en consola todo el proceso

    #nota: una optimizacion posible es hacer modelo.fit(X[index_train],Y[index_train]) y lo mismo en test para no ocupar tanta memoria
    modelo.fit(X_train,Y_train)

    ratio_aciertos = modelo.score(X_test,Y_test)
    print(f'Ratio de aciertos = {ratio_aciertos}')
    vector_aciertos =np.append(vector_aciertos,ratio_aciertos)
    

    #print(f"El ratio_aciertos es: {ratio_aciertos}")


#calculamos la media y varianza con ratios obtenidos
N=len(vector_aciertos)
media_ratio_aciertos= np.sum(vector_aciertos)/N
varianza_ratio_aciertos= np.sum((vector_aciertos - media_ratio_aciertos)**2)/N
print(f"Ratios obtenidos= {vector_aciertos}")
print(f"Media del ratio de aciertos= {media_ratio_aciertos} |||| Varianza del ratio de aciertos= {varianza_ratio_aciertos}")


######### utilizando KFold con 10 particiones ###################################
print("############################### Utilizando KFold con 10 particiones ############################")
fk= KFold(n_splits=10)#lo dejamos con shufle en false

vector_aciertos=np.array([])
for i,(index_train,index_test) in enumerate(fk.split(X)):
    print(f"Fold= {i}")

    X_train = X[index_train]
    Y_train = Y[index_train]

    X_test = X[index_test]
    Y_test = Y[index_test]

    #definimos el modelo dentro del for para entrenar el mismo modelo con cada fold
    modelo = MLPClassifier(hidden_layer_sizes=(100,80),activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                       random_state=42, tol=tol, verbose=False) #para que no muestre en consola todo el proceso

    #nota: una optimizacion posible es hacer modelo.fit(X[index_train],Y[index_train]) y lo mismo en test para no ocupar tanta memoria
    modelo.fit(X_train,Y_train)

    ratio_aciertos = modelo.score(X_test,Y_test)
    print(f'Ratio de aciertos = {ratio_aciertos}')
    vector_aciertos =np.append(vector_aciertos,ratio_aciertos)
    

    #print(f"El ratio_aciertos es: {ratio_aciertos}")


#calculamos la media y varianza con ratios obtenidos
N=len(vector_aciertos)
media_ratio_aciertos= np.sum(vector_aciertos)/N
varianza_ratio_aciertos= np.sum((vector_aciertos - media_ratio_aciertos)**2)/N
print(f"Ratios obtenidos= {vector_aciertos}")
print(f"Media del ratio de aciertos= {media_ratio_aciertos} |||| Varianza del ratio de aciertos= {varianza_ratio_aciertos}")

