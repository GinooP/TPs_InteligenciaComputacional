import numpy as np
import pandas as pd
import time as tm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# Lectura de Datos
tabla_combates_individuales = np.loadtxt('TP_Creativo/combates_indv_completo.csv',delimiter=',')

X_individuales = tabla_combates_individuales[:,:-1]
Y_individuales = tabla_combates_individuales[:,-1]
# print(X_individuales_aux)

X_indv_train_aux, X_indv_test, Y_indv_train, Y_indv_test = train_test_split(X_individuales, Y_individuales, test_size=0.25, random_state=42)

#invertimos y concatenamos los datos del train
# [0:24] primer pokemon, [24:48] segundo pokemon
X_indv_train_invertido= np.zeros(X_indv_train_aux.shape)

X_indv_train_invertido[:,0:24] = X_indv_train_aux[:,24:48]
X_indv_train_invertido[:,24:48] = X_indv_train_aux[:,0:24]

X_indv_train_aux= np.concatenate((X_indv_train_aux,X_indv_train_invertido),axis=0)#concatena en por filas


Y_indv_train_invertido= 1 - Y_indv_train # si Y=0 queda 1 y si Y=1 queda 0
Y_indv_train = np.concatenate((Y_indv_train,Y_indv_train_invertido),axis=0)

X_indv_train_aux, Y_indv_train = shuffle(X_indv_train_aux, Y_indv_train, random_state=42) #desordenamos los datos


diff_caracteristicas = X_indv_train_aux[:,0:6] - X_indv_train_aux[:,24:30]

X_indv_train = np.concatenate((diff_caracteristicas,X_indv_train_aux[:,6:24],X_indv_train_aux[:,30:48]),axis=1)


print(X_indv_train.shape)


fk = KFold(n_splits=3)

######################### busqueda en arbol #########################################
################################################### Arbol de decisión ###############################################################
# parametros = [
#     {
#         'criterion':['gini','entropy', 'log_loss'],
#         'min_impurity_decrease': [0.0,0.01,0.1,0.05],
        
#     }
# ]

# busqueda = GridSearchCV(
#     estimator=DecisionTreeClassifier(),
#     param_grid=parametros,
#     cv=fk,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=2
# )

# busqueda.fit(X_indv_train, Y_indv_train)

# print("Mejores hiperparámetros:")
# print(busqueda.best_params_)

# print("Mejor accuracy:")
# print(busqueda.best_score_)

###################### MUlticapa ############################
#################### Busqueda de hiperparametros en MLPC ################################

# parametros = [
#     {
#         'hidden_layer_sizes': [(10,10),(10,20),(10,40),(10,80),(10,100),(10,160),
#                                (20,10),(20,20),(20,40),(20,80),(20,100),(20,160),
#                                (40,10),(40,20),(40,40),(40,80),(40,100),(40,160),
#                                (80,10),(80,20),(80,40),(80,80),(80,100),(80,160),
#                                (100,10),(100,20),(100,40),(100,80),(100,100),(100,160)],
#         'activation': ['relu','logistic'],
#         'solver':['adam','sgd'],
#         'learning_rate_init':[0.01,0.1],
#         'learning_rate':['adaptive'],
#         'max_iter':[200],
#         'n_iter_no_change':[100] 
#     }
# ]

# busqueda = GridSearchCV(
#     estimator=MLPClassifier(),
#     param_grid=parametros,
#     cv=fk,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=2
# )

# busqueda.fit(X_indv_train, Y_indv_train)

# print("Mejores hiperparámetros:")
# print(busqueda.best_params_)

# print("Mejor accuracy:")
# print(busqueda.best_score_)
#########################################################################
# # Constantes MLP
eta = 0.05
epocas = 1000
tol = 0.95
capas=(64,32)


vector_aciertos_MLP = []
vector_aciertos_AD = []
vector_aciertos_SVC = []

for i,(index_train,index_test) in enumerate(fk.split(X_indv_train)):
    print(f'Kfol = {i}')

    # --- 1. Perceptrón Multicapa (MLP) ---
    tini = tm.perf_counter()
    modelo = MLPClassifier(hidden_layer_sizes=capas,activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                            tol=tol, verbose=False) #para que no muestre en consola todo el proceso
    
    modelo.fit(X_indv_train[index_train],Y_indv_train[index_train])
    vector_aciertos_MLP.append(modelo.score(X_indv_train[index_test],Y_indv_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion MLP = {tfin - tini} s')

    # --- 2. Arbol de decisión (AD) ---
    tini = tm.perf_counter()
    tree = DecisionTreeClassifier(criterion='gini',splitter='best')

    tree.fit(X_indv_train[index_train],Y_indv_train[index_train])
    vector_aciertos_AD.append(tree.score(X_indv_train[index_test],Y_indv_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion Árbol de Decisión = {tfin - tini} s')

    
    # --- 3. Máquina de soporte vectorial (SVC) ---
    # tini = tm.perf_counter()
    # clasificador_SVC = SVC(kernel='rbf', C=1.0)

    # clasificador_SVC.fit(X_indv_train[index_train], Y_indv_train[index_train])
    # vector_aciertos_SVC.append(clasificador_SVC.score(X_indv_train[index_test], Y_indv_train[index_test]))
    # tfin = tm.perf_counter()
    # print(f'Duracion SVC = {tfin - tini} s')

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

# print('\n--- Máquina de soporte vectorial (SVC) ---')
# media_mlp = np.sum(vector_aciertos_SVC)/N
# varianza_mlp = np.sum((vector_aciertos_SVC - media_mlp)**2)/N
# print(f"Ratios obtenidos = {vector_aciertos_SVC}")
# print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

