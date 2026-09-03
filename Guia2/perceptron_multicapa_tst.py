import numpy as np
import matplotlib.pyplot as plt

class Capa:
    def __init__(self, W, y, delta):
        self.W = W # Matriz de pesos
        self.y = y # Vector de salidas
        self.delta = delta # Vector de errores


def test_perceptron_multicapa(capas,vector_capas, patrones, winout):
    M = len(patrones[0,:]) - capas[-1] + 1# cantidad de entradas + bias
    N = len(patrones[:,0]) # cantidad de patrones

    # Matriz de Entradas
    x = -1 * np.ones((N,M))
    x[:,1:M] = patrones[:,0:M-1]
    #print(x)
    

    # Vector de salidas esperadas
    d = patrones[:,M-1:len(patrones[0,:])]#concideramos que puede haber mas salidas
    #print(d)

    # Definición de la función anónima con lambda
    sigmoide = lambda x: 2 / (1 + np.exp(-x)) - 1

    cant_capas = len(vector_capas)

    aciertos=0
    error_cuadratico=0
    for j in range(N):
        #obtenemos la salida lineal de la primer capa
        #cambiamos la funcion sigmoidea por la signo para para la verificación
        z = vector_capas[0].W @ x[j,:] 
        y = sigmoide(z)
        y = np.insert(y, 0, -1)
        vector_capas[0].y = y

        for k in range(1, cant_capas):
            z = vector_capas[k].W @ vector_capas[k-1].y 
            y = sigmoide(z)
            y = np.insert(y, 0, -1)
            vector_capas[k].y = y

        #print(vector_capas[-1].y)
        error_cuadratico += np.sum((d[j] - vector_capas[-1].y[-1])**2)
        if(winout):
            pos_mayor=np.argmax(vector_capas[-1].y)
            vector_capas[-1].y= -1* np.ones(np.size(vector_capas[-1].y))
            vector_capas[-1].y[pos_mayor]=1
            #print(f"este es el vector de capas del final: {vector_capas[-1].y}")
        else:
            vector_capas[-1].y = np.sign(vector_capas[-1].y)

        #print(f"este es el vector de salidas correctas yd={d[j,:]}")
        #print(f"{vector_capas[-1].y[1:]}")
        if(np.array_equal(vector_capas[-1].y[1:],d[j,:])):
            aciertos += 1

    #print(vector_capas[-1].y)
    error_cuadratico= error_cuadratico/N
    ratio_aciertos=aciertos/N #aciertos/cantidad de patrones
    #print(f"ratio de aciertos: {ratio_aciertos} epoca: {i}")
    print(f"ratio de aciertos: {ratio_aciertos} |||| error cuadratico: {error_cuadratico}")

    return vector_capas

