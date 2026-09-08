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
 
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.25,random_state=42) # shuffle=True


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
