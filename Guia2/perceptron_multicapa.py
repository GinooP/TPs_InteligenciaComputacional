import numpy as np
import matplotlib.pyplot as plt

class Capa:
    def __init__(self, W, y, delta):
        self.W = W # Matriz de pesos
        self.y = y # Vector de salidas
        self.delta = delta # Vector de errores


def generar_perceptron_multicapa(capas, patrones, eta, epocas, tol):
    # capas: vector con nro de neuronas en cada capa (ej: [2 3] -> 2 capas con 2 neuronas en la primera y 3 en la segunda)
    # patrones: matriz con los patrones (entradas y salidas esperadas)
    # eta: tasa de aprendizaje
    # epocas: cantidad de etapas de aprendizaje
    # tol: error aceptado/ratio de aciertos buscado

    M = len(patrones[0,:]) - capas[-1] + 1# cantidad de entradas + bias
    N = len(patrones[:,0]) # cantidad de patrones

    # Matriz de Entradas
    x = -1 * np.ones((N,M))
    x[:,1:M] = patrones[:,0:M-1]
    #print(x)
    

    # Vector de salidas esperadas
    print(capas[-1])
    d = patrones[:,M-1:len(patrones[0,:])]#concideramos que puede haber mas salidas
    #print(d)


    # 1. Inicialización aleatoria
    # Inicialización de los vectores W para cada capa
    cant_capas = len(capas)
    vector_capas = []

    rng = np.random.default_rng()
    W = np.array(rng.random((capas[0], M)) - 0.5) #  Setear los pesos entre [-0.5 0.5] del vector de pesos sináptico
    vector_capas.append(Capa(W, [], []))
    for i in range(1,cant_capas):
        W = np.array(rng.random((capas[i], capas[i-1]+1)) - 0.5) #  Setear los pesos entre [-0.5 0.5] del vector de pesos sináptico
        vector_capas.append(Capa(W, [], []))

    # Definición de la función anónima con lambda
    sigmoide = lambda x: 2 / (1 + np.exp(-x)) - 1

    numero_epoca=0
    error_cuadratico=0
    ratio_aciertos=0
    for i in range(epocas):
        for j in range(N):

            # 2. Propagación hacia adelante
            z = vector_capas[0].W @ x[j,:] 
            y = sigmoide(z)
            y = np.insert(y, 0, -1)
            vector_capas[0].y = y

            for k in range(1, cant_capas):
                z = vector_capas[k].W @ vector_capas[k-1].y 
                y = sigmoide(z)
                y = np.insert(y, 0, -1)
                vector_capas[k].y = y
                # print(vector_capas[k].y)

            # 3. Propagación hacia atras
            sizey= len(y)
            delta = (d[j,:] - y[1:sizey]) * 0.5 * (1 + y[1:sizey]) * (1 - y[1:sizey])
            vector_capas[-1].delta = delta
            #print(delta)

            for k in range(2, cant_capas + 1):
                #print(vector_capas[-k + 1].W[:,1:])
                #print(vector_capas[-k + 1].delta)
                delta = (vector_capas[-k + 1].delta @ vector_capas[-k + 1].W[:,1:]) * 0.5 * (1 + vector_capas[-k].y[1:]) * (1 - vector_capas[-k].y[1:])
                vector_capas[-k].delta = delta
                # print(delta)
            
            # 4. Adaptación de los pesos
            #print(x[j,:])
            #print(vector_capas[0].delta)
            variacion_pesos = eta*np.outer(vector_capas[0].delta, x[j,:])
            #print(variacion_pesos)
            vector_capas[0].W=vector_capas[0].W + variacion_pesos
            for k in range(1,cant_capas-1):
                #variacion_pesos = eta * np.dot(vector_capas[k+1].delta,vector_capas[k+1].W) * (1 + vector_capas[k].y[1:])*(1 - vector_capas[k].y[1:]) * vector_capas[k-1].y
                #print(vector_capas[k-1].y)
                #print(vector_capas[k].delta)
                #print(vector_capas[k].delta @ vector_capas[k-1].y)
                variacion_pesos = eta*np.outer(vector_capas[k].delta, vector_capas[k-1].y)
                vector_capas[k].W = vector_capas[k].W + variacion_pesos

        # 5. Iteración: vuelve a 2 hasta convergencia o finalización
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
            error_cuadratico += (d[j] - vector_capas[-1].y[-1])**2
            if(np.sign(vector_capas[-1].y[-1]) == d[j,:]):
                aciertos += 1

        #print(vector_capas[-1].y)
        error_cuadratico= error_cuadratico/N
        ratio_aciertos=aciertos/N #aciertos/cantidad de patrones
        #print(f"ratio de aciertos: {ratio_aciertos} epoca: {i}")
        print(f"ratio de aciertos: {ratio_aciertos} |||| error cuadratico: {error_cuadratico} ||||| epoca: {i}")
        numero_epoca=i

        if (ratio_aciertos==1):
            print(f"converge en la epoca: {i}")
            break
    print(f"ratio de aciertos: {ratio_aciertos} |||| error cuadratico: {error_cuadratico} ||||| epoca: {i}")
            


# 5. Iteración: vuelve a 2 hasta convergencia o finalización

    
