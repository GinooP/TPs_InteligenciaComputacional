# arquitectura para una red recurrente de Hopfield (mem. asociativa) con su entrenamiento hebbiano

import numpy as np

def red_hopfield(patrones):
    # patrones: matriz de MxN donde M es la cantidad de patrones y N es la dimension de los patrones

    # Almacenamiento (Entrenamiento)
    M = len(patrones[:,0]) # Cantidad de patrones
    N = len(patrones[0,:]) # Dimensión de los patrones

    W = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if i != j:
                W[i,j] = (1/N)*np.dot(patrones[:,i], patrones[:,j])

    return W

def recuperar_rh(patron, W):
    # Recuperacion (Prueba de la Red de Holfield)
    y_n1 = patron 
    N = len(patron)

    rng = np.random.default_rng()

    converge = False
    while not converge:
        j = rng.integers(low=0, high=N)
        y = 
    ...
W = red_hopfield(np.ones((10,10)))

recuperar_rh([1,2,3,4,5],0)