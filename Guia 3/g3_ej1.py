import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
])

y = np.array([10, 20, 30, 40, 50, 60, 70, 80])

#train_test_split me separa los patrones (ya divididos en entradas y salidas) en un conjunto para entrenamiento y otro para pruebas
#con test_size controlamos que proporcion de los datos van para el test
# con test_size=0.25 el 25% va para el test y otro 75% para el entrenamiento
# tambien se usar train_size de la misma forma (si se omite uno de los dos el otro se calcula en base al disponible)
#ramdom_state permite reproducir la distribucion aleatoria de los patrones siempre que no se omita su valor
# si a test_size le doy un valor entero (test_size=2) indica que voy a tener exactamente dos muestras en la prueba
# y el resto de las muestras van a estar en el entrenamiento
 
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.25,shuffle=True)


print(f"X_train: {X_train}")
print(f"X_test: {X_test}")
print(f"Y_train: {Y_train}")
print(f"Y_test {Y_test}")


############## KFold ##############################

#shufle true indica si queremos que mezcle aleatoriamente los datos antes de hacer las diviciones
kf=KFold(n_splits=4,shuffle=False)
#cuando llamemos al metodo split de la clase esta va a hacer 4 divisiones los patrones
#de las cuales una se va a utilziar para el test y el resto para el entrenamiento
#sin embargo la division no se realiza realmente, si no que se hace a través de índices

for i,(index_train , index_test) in enumerate(kf.split(X)):

    #al usarlo de esta forma usamos una vez cada conjunto armado para el test
    # primero usamos el primero, despues el segundo conjunto y así
     
    print(f"####################### Fold= {i} ##############################")
    #patrones para el entrenamiento
    X_train= X[index_train]
    Y_train= y[index_train]

    #patrones para el test
    X_test= X[index_test]
    Y_test= y[index_train]

    print(f"X_train= {X_train}")
    print(f"X_test= {X_test}")

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
tol=1e-4
modelo = MLPClassifier(hidden_layer_sizes=(100,80),activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                       random_state=42,tol=tol,verbose=True)

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

    ratio_aciertos= modelo.score(X_test,Y_test)
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

    ratio_aciertos= modelo.score(X_test,Y_test)
    vector_aciertos =np.append(vector_aciertos,ratio_aciertos)
    

    #print(f"El ratio_aciertos es: {ratio_aciertos}")


#calculamos la media y varianza con ratios obtenidos
N=len(vector_aciertos)
media_ratio_aciertos= np.sum(vector_aciertos)/N
varianza_ratio_aciertos= np.sum((vector_aciertos - media_ratio_aciertos)**2)/N
print(f"Ratios obtenidos= {vector_aciertos}")
print(f"Media del ratio de aciertos= {media_ratio_aciertos} |||| Varianza del ratio de aciertos= {varianza_ratio_aciertos}")

