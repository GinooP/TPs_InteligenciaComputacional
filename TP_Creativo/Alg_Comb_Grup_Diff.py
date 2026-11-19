import numpy as np
import pandas as pd
import time as tm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.svm import SVC


# Lectura de Datos
tabla_combates_grupales = np.loadtxt('TP_Creativo/combates_grup_completo.csv',delimiter=',')

X_grupales = tabla_combates_grupales[:,:-1]
Y_grupales = tabla_combates_grupales[:,-1]

X_grup_train, X_grup_test, Y_grup_train, Y_grup_test = train_test_split(X_grupales, Y_grupales, test_size=0.25, random_state=42)

##### duplicamos los datos del train e invertimos los equipos de la segunda mitad por si hay sesgo
# 0:144 es el primer el equipo (el rango llega hasta 143) y 144:288 es el segundo equipo
X_grup_train_invertido= np.zeros(X_grup_train.shape)

X_grup_train_invertido[:,0:144] = X_grup_train[:,144:288]
X_grup_train_invertido[:,144:288] = X_grup_train[:,0:144]

X_grup_train = np.concatenate((X_grup_train,X_grup_train_invertido),axis=0)#concatena en por filas

Y_grup_train_invertido= 1 - Y_grup_train # si Y=0 queda 1 y si Y=1 queda 0

Y_grup_train = np.concatenate((Y_grup_train,Y_grup_train_invertido),axis=0)

X_grup_train, Y_grup_train = shuffle(X_grup_train, Y_grup_train, random_state=42) #desordenamos los datos

# 0:6 - 24:30 - 48:54 - 72:78 - 96:102 - 120:126
prom_eq_1 = (X_grup_train[:,0:6] + X_grup_train[:,24:30] + X_grup_train[:,48:54] + X_grup_train[:,72:78] + X_grup_train[:,96:102] + X_grup_train[:,120:126] ) / 6

type_eq_1 = X_grup_train[:,6:24] + X_grup_train[:,30:48] + X_grup_train[:,54:72] + X_grup_train[:,78:96] + X_grup_train[:,102:120] + X_grup_train[:,126:144]
# 144:150 - 168:174 - 192:198 - 216:222 - 240:246 - 264:270
prom_eq_2 = (X_grup_train[:,144:150] + X_grup_train[:,168:174] + X_grup_train[:,192:198] + X_grup_train[:,216:222] + X_grup_train[:,240:246] + X_grup_train[:,264:270] ) / 6

type_eq_2 = X_grup_train[:,150:168] + X_grup_train[:,174:192] + X_grup_train[:,198:216] + X_grup_train[:,222:240] + X_grup_train[:,246:264] + X_grup_train[:,270:288]

diff_prom_stats = prom_eq_1 - prom_eq_2
# print(diff_prom_stats)

X_grup_train = np.concatenate((diff_prom_stats,type_eq_1,type_eq_2),axis=1)
# print(X_grup_train)
# print(X_grup_train.shape)


fk = KFold(n_splits=3)

################################## Busqueda usando gridsearch ###########################################################################
# fk = KFold(n_splits=3)

# parametros = [
#     {
#         'kernel': ['poly'],
#         'C': [0.01, 0.1, 1, 5, 10],
#         'gamma': [0.1, 0.5, 1, 0.01],
#         'degree': [2, 3, 4, 5]
#     },
#     {
#         'kernel': ['sigmoid', 'rbf'],
#         'C': [0.01, 0.1, 1, 5, 10],
#         'gamma': [0.1, 0.5, 1, 0.01]
#     }
# ]

# busqueda = GridSearchCV(
#     estimator=SVC(),
#     param_grid=parametros,
#     cv=fk,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=2
# )

# busqueda.fit(X_grup_train, Y_grup_train)

# print("Mejores hiperparámetros:")
# print(busqueda.best_params_)

# print("Mejor accuracy:")
# print(busqueda.best_score_)

#################### Busqueda de hiperparametros en MLPC ################################

# parametros = [
#     {
#         'hidden_layer_sizes': [(20,40),(100,50)],
#         'activation': ['relu'],
#         'learning_rate_init':[0.001,0.01,0.1,0.05,0.5],
#         'learning_rate':['adaptive'],
#         'max_iter':[500,200,750,250,1000],
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

# busqueda.fit(X_grup_train, Y_grup_train)

# print("Mejores hiperparámetros:")
# print(busqueda.best_params_)

# print("Mejor accuracy:")
# print(busqueda.best_score_)

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

# busqueda.fit(X_grup_train, Y_grup_train)

# print("Mejores hiperparámetros:")
# print(busqueda.best_params_)

# print("Mejor accuracy:")
# print(busqueda.best_score_)

# arbol=DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0.0)
# arbol.fit(X_grup_train,Y_grup_train)
# print(arbol.get_depth())
############################# busqueda de hiperparametros a pata ###################################################
# # rango de constantes de SVM

# kernel=['poly','sigmoid','rbf']
# C=[0.01,0.1,1,5,10]
# gamma=[0.1,0.5,1,0.01]
# degree=[2,3,4,5]

# mejor_kernel=0
# mejor_c=0
# mejor_gamma=0
# mejor_degree=0
# mejor_media_ratio_aciertos=0
# print(len(kernel))

# #buscamos la mejor convinacion de hiperparametros para SVM
# for i in range(len(kernel)):
#     for j in range(len(C)):
#         for k in range(len(gamma)):
            
#             if(kernel[i]=='poly'):
#                 for l in range(len(degree)):
#                     vector_aciertos_SVC = []
#                     for m,(index_train,index_test) in enumerate(fk.split(X_grup_train)):
#                         # --- 3. Máquina de soporte vectorial (SVC) ---
#                         #tini = tm.perf_counter()
#                         clasificador_SVC = SVC(kernel=kernel[i], C=C[j],gamma=gamma[k],degree=degree[l])

#                         clasificador_SVC.fit(X_grup_train[index_train], Y_grup_train[index_train])
#                         vector_aciertos_SVC.append(clasificador_SVC.score(X_grup_train[index_test], Y_grup_train[index_test]))

#                     N=len(vector_aciertos_SVC)
#                     media_svm = np.sum(vector_aciertos_SVC)/N
#                     print(f"kernel={kernel[i]} | C={C[j]} | gamma={gamma[k]} | degree={degree[l]} | ratio={media_svm}")

#                     # si la media es mejor guardamos la configuracion
#                     if(media_svm > mejor_media_ratio_aciertos):
#                         mejor_media_ratio_aciertos=media_svm
#                         mejor_c= C[j]
#                         mejor_kernel=kernel[i]
#                         mejor_gamma=gamma[k]
#                         mejor_degree=degree[l]
#             else:
#                 vector_aciertos_SVC = []
#                 for m,(index_train,index_test) in enumerate(fk.split(X_grup_train)):
#                     # --- 3. Máquina de soporte vectorial (SVC) ---
#                     #tini = tm.perf_counter()
#                     clasificador_SVC = SVC(kernel=kernel[i], C=C[j],gamma=gamma[k])

#                     clasificador_SVC.fit(X_grup_train[index_train], Y_grup_train[index_train])
#                     #tfin = tm.perf_counter()
#                     vector_aciertos_SVC.append(clasificador_SVC.score(X_grup_train[index_test], Y_grup_train[index_test]))

#                 N=len(vector_aciertos_SVC)
#                 media_svm = np.sum(vector_aciertos_SVC)/N
#                 print(f"kernel={kernel[i]} | C={C[j]} | gamma={gamma[k]} | ratio={media_svm}")

#                 # si la media es mejor guardamos la configuracion
#                 if(media_svm > mejor_media_ratio_aciertos):
#                     mejor_media_ratio_aciertos=media_svm
#                     mejor_c= C[j]
#                     mejor_kernel=kernel[i]
#                     mejor_gamma=gamma[k]

# print("##################### Mejores Hiperparametros ############################################")
# print(f"kernel={mejor_kernel} | C={mejor_c} | gamma={mejor_gamma} | degree={mejor_degree} | ratio={mejor_media_ratio_aciertos}")

######################################################################
# # Constantes MLP
eta = 0.01
epocas = 500
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
                            tol=tol, verbose=False,n_iter_no_change=100,learning_rate='adaptive') #para que no muestre en consola todo el proceso
    
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

