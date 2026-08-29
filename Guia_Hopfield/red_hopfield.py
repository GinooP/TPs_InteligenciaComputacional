# arquitectura para una red recurrente de Hopfield (mem. asociativa) con su entrenamiento hebbiano

import numpy as np

def red_hopfield(patrones):
    # patrones: matriz de MxN donde M es la cantidad de patrones y N es la dimension de los patrones

    # Almacenamiento (Entrenamiento)
    M = len(patrones[:,0]) # Cantidad de patrones
    N = len(patrones[0,:]) # Dimensión de los patrones

    # dimension = int(np.sqrt(N))
    # print(f'Cantidad de patrones máximos para esta red de {dimension}x{dimension} pixeles: {int(N/(2*np.log(N)))}')

    W = (1/N) * (patrones.T @ patrones)
    
    # Anulamos la diagonal principal
    np.fill_diagonal(W, 0)

    # W = np.zeros((N,N))
    # for i in range(N):
    #     for j in range(N):
    #         if i != j:
    #             W[i,j] = (1/N)*np.dot(patrones[:,i], patrones[:,j])

    return W

def recuperar_rh(patron, W):
    # Recuperacion (Prueba de la Red de Holfield)
    y = patron 
    N = len(patron)

    historial_estados = []
    historial_estados.append(y.copy())

    estable = False
    epoca = 0
    while not estable:
        epoca += 1
        cambios = 0 # Detectar cambios en la epoca

        indices = np.random.permutation(N) # Recorrer las neuronas en un orden aleatorio

        for j in indices:

            x = np.dot(W[j,:], y) # Calculo de la sumatoria (relacion en el pixel j)
            nuevo_estado = np.sign(x) 

            if nuevo_estado != y[j]: # Se detectó un cambio
                cambios += 1
                y[j] = nuevo_estado # Actualizo el estado
                historial_estados.append(y.copy()) # Guardo la iteracion n


        if cambios == 0: # Si no hay cambios recorriendo todas las neuronas, finalizar
            estable = True
    historial = np.array(historial_estados)
    return y, historial, epoca

def agregar_ruido(patron, probabilidad):

    # 1. Hacemos una copia para no pisar el patrón original en memoria
    patron_ruidoso = patron.copy()
    
    # 2. Generamos un vector de números aleatorios uniformes [0, 1) del mismo tamañoaleatorios = np.random.rand(len(patron))
    aleatorios = np.random.rand(len(patron))
    
    # 3. Creamos una máscara booleana donde se cumple la condición.
    # Ej: si probabilidad es 0.1, estadísticamente el 10% de los valores serán True
    pixeles_a_invertir = aleatorios < probabilidad
    
    # 4. Multiplicamos por -1 solo los píxeles indicados por la máscara
    patron_ruidoso[pixeles_a_invertir] *= -1
    
    return patron_ruidoso