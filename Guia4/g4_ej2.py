import numpy as np
import k_means as km
import SOM as s

patrones_trn = np.loadtxt('Guia2/iris81_trn.csv', delimiter=',', skiprows=0)

# K medias
patrones = patrones_trn[:,0:4]
nro_grupos = 3
epocas = 100
K, lotes_patrones = km.k_means(patrones, nro_grupos, epocas)

# SOM
dim_matriz_neuronas = [10, 10]
eta = 0.9
vecindad = 3
cuadrado = False
epocas = [100, 200, 300]
# matriz_neuronas = s.som(patrones, dim_matriz_neuronas, epocas, eta, vecindad, cuadrado, graf=False)

# Matriz de Contingencia

# Comparacion con etiquetas reales, de referencia, clases conocidas, etc. ´
# Comparacion entre diferentes soluciones de clustering. ´
# Matriz de contingencia:
entradas = patrones_trn[:,0:4]
salidas = patrones_trn[:,4:7]
# print(entradas)
# print(salidas)
matriz_contingencia = km.tst_k_means(K, lotes_patrones, entradas, salidas)
print(matriz_contingencia)