import numpy as np
import matplotlib.pyplot as plt

class Capa:
    def __init__(self, W, y, delta):
        self.W = W # Matriz de pesos
        self.y = y # Vector de salidas
        self.delta = delta # Vector de errores


def generar_perceptron_multicapa(capas, patrones, eta, epocas, tol, one_hot=False):
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
    errores_de_clasificacion = []
    
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
        # error_acum = 0
        # aciertos = 0
        # error_de_clasificacion = 0
        # for j in range(N):
        #     z = vector_capas[0].W @ x[j,:]
        #     y = sigmoide(z)
        #     y = np.insert(y, 0, -1)
        #     vector_capas[0].y = y

        #     for k in range(1, cant_capas):
        #         z = vector_capas[k].W @ vector_capas[k-1].y 
        #         y = sigmoide(z)
        #         y = np.insert(y, 0, -1)
        #         vector_capas[k].y = y

        #     error_acum += np.sum((y[1:] - d[j])**2)

        #     salida = vector_capas[-1].y[1:]
        #     if (one_hot):
        #         imax = np.argmax(salida)
        #         salida[:] = -1
        #         salida[imax] = 1
        #     else:
        #         salida = np.sign(salida)

        #     if (np.array_equal(salida, d[j])):
        #         aciertos += 1
        #     else:
        #         error_de_clasificacion += 1

        # # 6. Testear porcentaje de aciertos (VECTORIZADO)
        # # Transponemos 'x' para operar sobre todos los patrones a la vez: forma (M, N)
        Y_eval = x.T 
        
        for k in range(cant_capas):
            Z = vector_capas[k].W @ Y_eval
            Y_eval = sigmoide(Z)
            # Agregamos la fila de bias (-1) en la parte superior para todos los patrones
            Y_eval = np.vstack([-np.ones((1, N)), Y_eval])
        
        # Extraemos la salida continua descartando el bias y transponiendo a forma (N, salidas)
        salida_continua = Y_eval[1:, :].T
        
        # Calculamos el error cuadrático acumulado de toda la matriz
        error_acum = np.sum((salida_continua - d)**2)
        
        # Discretización matricial
        if one_hot:
            imax = np.argmax(salida_continua, axis=1)
            salida_discreta = -np.ones_like(salida_continua)
            salida_discreta[np.arange(N), imax] = 1
        else:
            salida_discreta = np.sign(salida_continua)
            
        # Cálculo de aciertos (compara fila por fila y suma los verdaderos)
        aciertos = np.sum(np.all(salida_discreta == d, axis=1))
        error_de_clasificacion = N - aciertos

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
        ratios.append(aciertos/N*100)
        errores_de_clasificacion.append(error_de_clasificacion)
        # print(f'Error = {error:.8f} | Ratio = {ratios[-1]:.2f} | Epoca = {i+1}')

        # 7. Verificar que sea mayor a la tolerancia
        if ratios[-1] >= tol:
            print(f'El perceptrón multicapa convergió con un ratio de {ratios[-1]:.2f} % en la epoca {i+1}.')
            return vector_capas, np.array(ratios), np.array(errores), np.array(errores_de_clasificacion)

    # plt.ioff()
    print(f'El perceptrón multicapa tuvo un ratio de {ratios[-1]:.2f} % en la epoca final ({epocas}) con una TA de {eta}.')
    return vector_capas, np.array(ratios), np.array(errores), np.array(errores_de_clasificacion)


def tst_perceptron_multicapa(patrones, vector_capas, capas, one_hot=False, plotear=False):

    M = len(patrones[0,:]) - capas[-1] + 1 # cantidad de entradas + bias
    N = len(patrones[:,0]) # cantidad de patrones

    # Matriz de Entradas
    x = -1 * np.ones((N,M))
    x[:,1:M] = patrones[:,0:M-1]

    # Matriz de salidas esperadas
    d = patrones[:,M-1::]

    # Inicialización de los vectores W para cada capa
    cant_capas = len(capas)

    # Definición de la función anónima con lambda
    sigmoide = lambda x: 2 / (1 + np.exp(-x)) - 1

    aciertos = 0
    error_acum = 0
    error_clasificacion = 0
    
    # Vector para almacenar las predicciones finales y usarlas en el gráfico
    predicciones = np.zeros((N, d.shape[1]))

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

        error_acum += np.sum((y[1:] - d[j])**2)
                    
        salida = vector_capas[-1].y[1:]
        if (one_hot):
            imax = np.argmax(salida)
            salida[:] = -1
            salida[imax] = 1
        else:
            salida = np.sign(salida)

        # Guardamos la predicción discretizada
        predicciones[j,:] = salida

        if (np.array_equal(salida, d[j])):
            aciertos += 1
        else:
            error_clasificacion += 1

    error = (1/N) * error_acum
    ratio = aciertos/N*100

    print(f'La prueba dió un resultado de: ratio = {ratio:.2f} % | error cuadrático = {error:.5f}. | errores de clasificacion = {error_clasificacion}')
    
    # BLOQUE DE GRAFICACIÓN PARA EL EJERCICIO 2
    if plotear:
        # Verificamos que el problema sea bidimensional (2 entradas + 1 bias)
        if M == 3: 
            plt.figure(figsize=(8, 8))
            
            # Separamos las coordenadas x e y de la matriz de entrada
            eje_x = x[:, 1]
            eje_y = x[:, 2]
            
            if not one_hot:
                # Extraemos la predicción y las etiquetas reales (aplanadas a 1D)
                preds = predicciones[:, 0]
                d_flat = d[:, 0] 
                
                # Filtros booleanos
                correctos = (preds == d_flat)
                incorrectos = (preds != d_flat)
                
                # 1. Aciertos Clase 1 (Ej: +1) - Cuadrados rojos
                idx_c1_ok = correctos & (preds == 1)
                plt.scatter(eje_x[idx_c1_ok], eje_y[idx_c1_ok], 
                            edgecolors='red', facecolors='none', marker='s', s=40, label='Acierto (+1)')
                
                # 2. Aciertos Clase 2 (Ej: -1) - Cruces negras
                idx_c2_ok = correctos & (preds == -1)
                plt.scatter(eje_x[idx_c2_ok], eje_y[idx_c2_ok], 
                            color='black', marker='x', s=40, label='Acierto (-1)')
                
                # 3. Muestras mal predichas - Puntos naranjas grandes
                plt.scatter(eje_x[incorrectos], eje_y[incorrectos], 
                            color='orange', marker='o', s=80, alpha=0.8, label='Clasificación Incorrecta')
            
            plt.title('Clasificación Final del Perceptrón Multicapa (con Errores)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.show()
        else:
            print("El gráfico 2D solo está disponible para datasets con exactamente 2 variables de entrada.")

    return ratio, error
    
def graficar_resultados(etas, lista_errores_cuadraticos, lista_errores_clas, num_ocultas):
    # Paleta de colores en tonos pastel/mate para los gráficos
    colores = ['#8ab0c1', '#d8a773', '#9ebf8f'] 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Error Cuadrático Total
    for i, eta in enumerate(etas):
        ax1.plot(lista_errores_cuadraticos[i], color=colores[i], label=f'eta = {eta}', linewidth=2)
    ax1.set_title(f'Error Cuadrático Total - {num_ocultas} Neurona(s) Oculta(s)')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Error Cuadrático')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Gráfico 2: Error de Clasificación
    for i, eta in enumerate(etas):
        ax2.plot(lista_errores_clas[i], color=colores[i], label=f'eta = {eta}', linewidth=2)
    ax2.set_title(f'Error de Clasificación - {num_ocultas} Neurona(s) Oculta(s)')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Error (Cantidad/Porcentaje)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()