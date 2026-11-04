import numpy as np
import pandas as pd
import time as tm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Lectura de Datos
tabla_combates_grupales = np.loadtxt('TP_Creativo/combates_grup_completo.csv',delimiter=',')

X_grupales = tabla_combates_grupales[:,:-1]
Y_grupales = tabla_combates_grupales[:,-1]

X_grup_train, X_grup_test, Y_grup_train, Y_grup_test = train_test_split(X_grupales, Y_grupales, test_size=0.25, random_state=42)

fk = KFold(n_splits=3)

# Constantes MLP
eta = 0.01
epocas = 100
tol = 0.95
capas = (64,32)


vector_aciertos_MLP = []
vector_aciertos_AD = []
vector_aciertos_SVC = []

for i,(index_train,index_test) in enumerate(fk.split(X_grup_train)):
    print(f'Kfol = {i}')

    # --- 1. Perceptrón Multicapa (MLP) ---
    tini = tm.perf_counter()
    modelo = MLPClassifier(hidden_layer_sizes=capas,activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                        random_state=42, tol=tol, verbose=False,n_iter_no_change=epocas+1,learning_rate='adaptive') #para que no muestre en consola todo el proceso
    
    modelo.fit(X_grup_train[index_train],Y_grup_train[index_train])
    vector_aciertos_MLP.append(modelo.score(X_grup_train[index_test],Y_grup_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion MLP = {tfin - tini} s')

    # --- 2. Arbol de decisión (AD) ---
    tini = tm.perf_counter()
    tree = DecisionTreeClassifier(criterion='gini',splitter='best')

    tree.fit(X_grup_train[index_train],Y_grup_train[index_train])
    vector_aciertos_AD.append(tree.score(X_grup_train[index_test],Y_grup_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion Árbol de Decisión = {tfin - tini} s')

    
    # --- 3. Máquina de soporte vectorial (SVC) ---
    tini = tm.perf_counter()
    clasificador_SVC = SVC(kernel='rbf', C=2.0)

    clasificador_SVC.fit(X_grup_train[index_train], Y_grup_train[index_train])
    vector_aciertos_SVC.append(clasificador_SVC.score(X_grup_train[index_test], Y_grup_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion SVC = {tfin - tini} s')

N = len(vector_aciertos_MLP)

print('\n--- Perceptrón Multicapa (MLP) ---')
media_mlp = np.sum(vector_aciertos_MLP)/N
varianza_mlp = np.sum((vector_aciertos_MLP - media_mlp)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_MLP}")
print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

print('\n--- Arbol de Decisión (AD) ---')
media_mlp = np.sum(vector_aciertos_AD)/N
varianza_mlp = np.sum((vector_aciertos_AD - media_mlp)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_AD}")
print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

print('\n--- Máquina de soporte vectorial (SVC) ---')
media_mlp = np.sum(vector_aciertos_SVC)/N
varianza_mlp = np.sum((vector_aciertos_SVC - media_mlp)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_SVC}")
print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

