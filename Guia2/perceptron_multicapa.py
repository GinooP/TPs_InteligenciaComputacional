import numpy as np

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

    M = len(patrones[0,:]) - capas[-1] + 1 # cantidad de entradas + bias
    N = len(patrones[:,0]) # cantidad de patrones

    # Matriz de Entradas
    x = -1 * np.ones((N,M))
    x[:,1:M] = patrones[:,0:M-1]
    # print(x)

    # Matriz de salidas esperadas
    d = patrones[:,M-1::]
    # print(d)

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
            delta = (d[j,:] - y[1:]) * 0.5 * (1 + y[1:]) * (1 - y[1:])
            vector_capas[-1].delta = delta
            #print(delta)

            for k in range(2, cant_capas+1):
                # print(vector_capas[-k + 1].W[:,1:])
                #print(vector_capas[-k + 1].delta)
                delta = (vector_capas[1-k].delta @ vector_capas[1-k].W[:,1:]) * 0.5 * (1 + vector_capas[-k].y[1:]) * (1 - vector_capas[-k].y[1:])
                vector_capas[-k].delta = delta
                # print(delta)

            # 4. Adaptación de los pesos
            # print(vector_capas[0].delta)
            deltaw = eta * np.outer(vector_capas[0].delta, x[j,:])
            vector_capas[0].W = vector_capas[0].W + deltaw

            for k in range(1, cant_capas):
                # Multiplicación de vector fila x vector columna = matriz de nxm
                deltaw = eta * np.outer(vector_capas[k].delta, vector_capas[k-1].y)
                # print(deltaw)
                vector_capas[k].W = vector_capas[k].W + deltaw

        # 5. Iteración: vuelve a 2 hasta convergencia o finalización

        # 6. Testear porcentaje de aciertos

        # 7. Verificar que sea mayor a la tolerancia

    return vector_capas
    
