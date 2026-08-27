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

    M = len(patrones[0,:]) - capas[-1] + 1 # cantidad de entradas + bias
    N = len(patrones[:,0]) # cantidad de patrones

    # Matriz de Entradas
    x = -1 * np.ones((N,M))
    x[:,1:M] = patrones[:,0:M-1]

    # Matriz de salidas esperadas
    d = patrones[:,M-1::]

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

    errores = []
    ratios = []
    
    # x_recta1 = np.linspace(min(x[:,1]) - 0.5, max(x[:,1]) + 0.5, 100)
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(epocas):

        # rng.shuffle(patrones)
        # x = -1 * np.ones((N,M))
        # x[:,1:M] = patrones[:,0:M-1]
        # d = patrones[:,M-1::]

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

            # 3. Propagación hacia atras
            delta = (d[j,:] - y[1:]) * 0.5 * (1 + y[1:]) * (1 - y[1:])
            vector_capas[-1].delta = delta

            for k in range(2, cant_capas+1):
                delta = (vector_capas[1-k].delta @ vector_capas[1-k].W[:,1:]) * 0.5 * (1 + vector_capas[-k].y[1:]) * (1 - vector_capas[-k].y[1:])
                vector_capas[-k].delta = delta

            # 4. Adaptación de los pesos
            deltaw = eta * np.outer(vector_capas[0].delta, x[j,:])
            vector_capas[0].W = vector_capas[0].W + deltaw

            for k in range(1, cant_capas):
                # Multiplicación de vector fila x vector columna = matriz de nxm
                deltaw = eta * np.outer(vector_capas[k].delta, vector_capas[k-1].y)
                vector_capas[k].W = vector_capas[k].W + deltaw

            

        # 5. Iteración: vuelve a 2 hasta convergencia o finalización

        # 6. Testear porcentaje de aciertos
        error_acum = 0
        aciertos = 0
        for j in range(N):
            z = vector_capas[0].W @ x[j,:]
            y = sigmoide(z)
            y = np.insert(y, 0, -1)
            vector_capas[0].y = y

            for k in range(1, cant_capas):
                z = vector_capas[k].W @ vector_capas[k-1].y 
                y = sigmoide(z)
                y = np.insert(y, 0, -1)
                vector_capas[k].y = y

            if (np.sign(y[1:]) == d[j]):
                aciertos += 1
            else:
                error_acum += np.sum((y[1:] - d[j])**2)

        # Grafico de las rectas para el XOR
        # y_recta1 = vector_capas[0].W[0,0]/vector_capas[0].W[0,2] - (vector_capas[0].W[0,1]/vector_capas[0].W[0,2])*x_recta1
        # y_recta2 = vector_capas[0].W[1,0]/vector_capas[0].W[1,2] - (vector_capas[0].W[1,1]/vector_capas[0].W[1,2])*x_recta1
        # ax.clear()
        # ax.scatter(x[:,1], x[:,2], c=d, cmap='bwr')
        # ax.plot(x_recta1, y_recta1, color='green', label='Neurona 1')
        # ax.plot(x_recta1, y_recta2, color='orange', label='Neurona 2')
        # ax.set_xlim(-1.5, 1.5) 
        # ax.set_ylim(-1.5, 1.5)
        # ax.set_title("XOR")
        # ax.grid(True, alpha=0.8)
        # plt.pause(0.3)

        error = (1/N) * error_acum
        errores.append(error)
        print(error)
        ratios.append(round(aciertos/N*100, 2))

        # 7. Verificar que sea mayor a la tolerancia
        if ratios[-1] >= tol:
            print(f'El perceptrón multicapa convergió con un ratio de {ratios[-1]:.0f} % en la epoca {i+1}.')
            return vector_capas, np.array(ratios), np.array(errores)

    # plt.ioff()
    print(f'Probar agregando más etapas. El perceptrón multicapa tuvo un ratio de {ratios[-1]:.0f} % en la epoca final.')
    return vector_capas, np.array(ratios), np.array(errores)


def tst_perceptron_multicapa(patrones, vector_capas, capas):

    M = len(patrones[0,:]) - capas[-1] + 1 # cantidad de entradas + bias
    N = len(patrones[:,0]) # cantidad de patrones

    # Matriz de Entradas
    x = -1 * np.ones((N,M))
    x[:,1:M] = patrones[:,0:M-1]

    # Matriz de salidas esperadas
    d = patrones[:,M-1::]

    # 1. Inicialización aleatoria
    # Inicialización de los vectores W para cada capa
    cant_capas = len(capas)

    # Definición de la función anónima con lambda
    sigmoide = lambda x: 2 / (1 + np.exp(-x)) - 1

    aciertos = 0
    error_acum = 0
    for j in range(N):
        z = vector_capas[0].W @ x[j,:]
        y = sigmoide(z)
        y = np.insert(y, 0, -1)
        vector_capas[0].y = y

        for k in range(1, cant_capas):
            z = vector_capas[k].W @ vector_capas[k-1].y 
            y = sigmoide(z)
            y = np.insert(y, 0, -1)
            vector_capas[k].y = y

        if (np.sign(y[1:]) == d[j]):
            aciertos += 1
        else:
            error_acum += np.sum((y[1:] - d[j])**2)

    error = (1/N) * error_acum
    ratio = round(aciertos/N*100, 2)

    print(f'La prueba dió un resultado de: ratio = {ratio:.0f} % | error cuadrático = {error}.')
    return ratio, error
    
