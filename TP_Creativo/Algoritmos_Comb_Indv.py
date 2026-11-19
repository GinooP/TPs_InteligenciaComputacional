import numpy as np
import pandas as pd
import time as tm
import matriz_de_confusion as mc

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



# Lectura de Datos
tabla_combates_individuales = np.loadtxt('TP_Creativo/combates_indv_completo.csv',delimiter=',')

X_indviduales = tabla_combates_individuales[:,:-1]
Y_indviduales = tabla_combates_individuales[:,-1]

X_indv_train, X_indv_test, Y_indv_train, Y_indv_test = train_test_split(X_indviduales, Y_indviduales, test_size=0.25, random_state=42)

#invertimos y concatenamos los datos del train
# [0:24] primer pokemon, [24:48] segundo pokemon
X_indv_train_invertido= np.zeros(X_indv_train.shape)

X_indv_train_invertido[:,0:24] = X_indv_train[:,24:48]
X_indv_train_invertido[:,24:48] = X_indv_train[:,0:24]

X_indv_train= np.concatenate((X_indv_train,X_indv_train_invertido),axis=0)#concatena en por filas


Y_indv_train_invertido= 1 - Y_indv_train # si Y=0 queda 1 y si Y=1 queda 0
Y_indv_train = np.concatenate((Y_indv_train,Y_indv_train_invertido),axis=0)

X_indv_train, Y_indv_train = shuffle(X_indv_train, Y_indv_train, random_state=42) #desordenamos los datos

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
#         'hidden_layer_sizes': [(80,10)],
#         'activation': ['relu','logistic'],
#         'solver':['adam','sgd'],
#         'learning_rate_init':[0.01,0.1,0.05,0.5],
#         'learning_rate':['adaptive'],
#         'max_iter':[200,500,750,1000],
#         'n_iter_no_change':[10] 
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
########################################################################
# Constantes MLP
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
    modelo = MLPClassifier(hidden_layer_sizes= (80,10), activation='relu', learning_rate='adaptive', learning_rate_init=0.01,
                            max_iter=1000, solver='sgd', n_iter_no_change=10) #para que no muestre en consola todo el proceso
    
    modelo.fit(X_indv_train[index_train],Y_indv_train[index_train])
    vector_aciertos_MLP.append(modelo.score(X_indv_train[index_test],Y_indv_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion MLP = {tfin - tini} s')

    # # --- 2. Arbol de decisión (AD) ---
    # tini = tm.perf_counter()
    # tree = DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0)

    # tree.fit(X_indv_train[index_train],Y_indv_train[index_train])
    # vector_aciertos_AD.append(tree.score(X_indv_train[index_test],Y_indv_train[index_test]))
    # tfin = tm.perf_counter()
    # print(f'Duracion Árbol de Decisión = {tfin - tini} s')

    
    # # --- 3. Máquina de soporte vectorial (SVC) ---
    # tini = tm.perf_counter()
    # clasificador_SVC = SVC(C= 15,gamma=0.05,kernel='rbf')

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

# print('\n--- Arbol de Decisión (AD) ---')
# media_mlp = np.sum(vector_aciertos_AD)/N
# varianza_mlp = np.sum((vector_aciertos_AD - media_mlp)**2)/N
# print(f"Ratios obtenidos = {vector_aciertos_AD}")
# print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

# print('\n--- Máquina de soporte vectorial (SVC) ---')
# media_mlp = np.sum(vector_aciertos_SVC)/N
# varianza_mlp = np.sum((vector_aciertos_SVC - media_mlp)**2)/N
# print(f"Ratios obtenidos = {vector_aciertos_SVC}")
# print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

###################### Determinar las matrices de confuncion #########################

# MLP: capas = (80,10), activation=”relu”, learning rate=”adaptive”, learning rate init=0.01, maxit=1000, solver=”sgd”, n iter no change=10
# SVM: ’C’ = 15, gamma = 0.05, kernel = ”rbf”
# DT: criterion=”gini”, min impurity decrease=0.0

multicapa= MLPClassifier(hidden_layer_sizes= (80,10), activation='relu', learning_rate='adaptive', learning_rate_init=0.01, max_iter=1000, solver='sgd', n_iter_no_change=10)

clasificador_SVC=SVC(C= 15,gamma=0.05,kernel='rbf')

tree = DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0)

multicapa.fit(X_indv_train,Y_indv_train)
#clasificador_SVC.fit(X_indv_train,Y_indv_train)
#tree.fit(X_indv_train,Y_indv_train)


#------------- multicapa -----------------------------------
Y_predicha=multicapa.predict(X_indv_test)#obtenemos los valores predichos para el test

matriz_confucion_MLP= mc.obtener_matriz_de_confusion(Y_indv_test,Y_predicha)

sensibilidad_eq1= matriz_confucion_MLP[0,0]/(matriz_confucion_MLP[0,0]+matriz_confucion_MLP[0,1])
precision_eq1= matriz_confucion_MLP[0,0]/(matriz_confucion_MLP[0,0] + matriz_confucion_MLP[1,0])
F1_eq1= 2*sensibilidad_eq1*precision_eq1/(sensibilidad_eq1 + precision_eq1)

sensibilidad_eq2= matriz_confucion_MLP[1,1]/(matriz_confucion_MLP[1,1] + matriz_confucion_MLP[1,0])
precision_eq2= matriz_confucion_MLP[1,1]/(matriz_confucion_MLP[1,1] + matriz_confucion_MLP[0,1])
F1_eq2= 2*sensibilidad_eq2*precision_eq2/(sensibilidad_eq2 + precision_eq2)

print("--------------- multicapa grupal mejores parametros -----------------------")
print("-------- matriz de confusión")
print(matriz_confucion_MLP)
print(f'ratio_aciertos={multicapa.score(X_indv_test,Y_indv_test)}')
print("-------medidas de desempeño haciendo foco en el equipo 1")
print(f"sensibilidad equipo 1 = {sensibilidad_eq1}")
print(f"precision equipo 1 = {precision_eq1}")
print(f"F1 equipo 1 = {F1_eq1}")
print(f"------medidas de desempeño haciendo foco en equipo 2")
print(f"sensibilidad equipo 2 = {sensibilidad_eq2}")
print(f"precision equipo 2 = {precision_eq2}")
print(f"F1 equipo 2 = {F1_eq2}")


# #------------ SVM ------------------------------------------
# Y_predicha=clasificador_SVC.predict(X_indv_test)#obtenemos los valores predichos para el test

# matriz_confucion_SVM= mc.obtener_matriz_de_confusion(Y_indv_test,Y_predicha)

# sensibilidad_eq1= matriz_confucion_SVM[0,0]/(matriz_confucion_SVM[0,0]+matriz_confucion_SVM[0,1])
# precision_eq1= matriz_confucion_SVM[0,0]/(matriz_confucion_SVM[0,0] + matriz_confucion_SVM[1,0])
# F1_eq1= 2*sensibilidad_eq1*precision_eq1/(sensibilidad_eq1 + precision_eq1)

# sensibilidad_eq2= matriz_confucion_SVM[1,1]/(matriz_confucion_SVM[1,1] + matriz_confucion_SVM[1,0])
# precision_eq2= matriz_confucion_SVM[1,1]/(matriz_confucion_SVM[1,1] + matriz_confucion_SVM[0,1])
# F1_eq2= 2*sensibilidad_eq2*precision_eq2/(sensibilidad_eq2 + precision_eq2)

# print("--------------- SVM grupal mejores parametros -----------------------")
# print("-------- matriz de confusión")
# print(matriz_confucion_SVM)
# print(f'ratio_aciertos={clasificador_SVC.score(X_indv_test,Y_indv_test)}')
# print("-------medidas de desempeño haciendo foco en el equipo 1")
# print(f"sensibilidad equipo 1 = {sensibilidad_eq1}")
# print(f"precision equipo 1 = {precision_eq1}")
# print(f"F1 equipo 1 = {F1_eq1}")
# print(f"------medidas de desempeño haciendo foco en equipo 2")
# print(f"sensibilidad equipo 2 = {sensibilidad_eq2}")
# print(f"precision equipo 2 = {precision_eq2}")
# print(f"F1 equipo 2 = {F1_eq2}")
# #------------ Árbol ----------------------------------------
# Y_predicha=tree.predict(X_indv_test)#obtenemos los valores predichos para el test

# matriz_confucion_tree= mc.obtener_matriz_de_confusion(Y_indv_test,Y_predicha)

# sensibilidad_eq1= matriz_confucion_tree[0,0]/(matriz_confucion_tree[0,0]+matriz_confucion_tree[0,1])
# precision_eq1= matriz_confucion_tree[0,0]/(matriz_confucion_tree[0,0] + matriz_confucion_tree[1,0])
# F1_eq1= 2*sensibilidad_eq1*precision_eq1/(sensibilidad_eq1 + precision_eq1)

# sensibilidad_eq2= matriz_confucion_tree[1,1]/(matriz_confucion_tree[1,1] + matriz_confucion_tree[1,0])
# precision_eq2= matriz_confucion_tree[1,1]/(matriz_confucion_tree[1,1] + matriz_confucion_tree[0,1])
# F1_eq2= 2*sensibilidad_eq2*precision_eq2/(sensibilidad_eq2 + precision_eq2)

# print("--------------- Árbol grupal mejores parametros -----------------------")
# print("-------- matriz de confusión")
# print(matriz_confucion_tree)
# print(f'ratio_aciertos={tree.score(X_indv_test,Y_indv_test)}')
# print("-------medidas de desempeño haciendo foco en el equipo 1")
# print(f"sensibilidad equipo 1 = {sensibilidad_eq1}")
# print(f"precision equipo 1 = {precision_eq1}")
# print(f"F1 equipo 1 = {F1_eq1}")
# print(f"------medidas de desempeño haciendo foco en equipo 2")
# print(f"sensibilidad equipo 2 = {sensibilidad_eq2}")
# print(f"precision equipo 2 = {precision_eq2}")
# print(f"F1 equipo 2 = {F1_eq2}")

