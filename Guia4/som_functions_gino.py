import numpy as np
import matplotlib.pyplot as plt
import time as tm
import SOM as sm

def som_mascara(patrones, dimension_mapa, epocas=[500, 1000, 3000], eta=0.9, vecindad=1, cuadrado=False, graf=False):

    filas, columnas = dimension_mapa
    cant_neuronas = filas * columnas
    cant_entradas = len(patrones[0,:])
    cant_etapas = len(epocas)

    rng = np.random.default_rng()

    # Vector con los pesos de cada neurona numerada desde 0 hasta cant_neuronas
    vec_neuronas = rng.random((cant_neuronas,cant_entradas))

    # Crear matriz de coordenadas 2D para el mapa topológico [fila, columna]
    X, Y = np.meshgrid(np.arange(columnas), np.arange(filas))
    coordenadas = np.c_[Y.ravel(), X.ravel()]

    m = (1-vecindad)/(cant_etapas-1) if cant_etapas > 1 else 0
    cant_vecinos = lambda x: int(m*x + vecindad) 

    a = eta
    b = np.log(0.1/eta) / (cant_etapas-1) if cant_etapas > 1 else 0
    valor_eta = lambda x: a*np.exp(b*x)

    for etapa in range(0,cant_etapas):

        cant_vecinos_etapa = cant_vecinos(etapa)
        valor_eta_etapa = valor_eta(etapa)
        for epoca in range(0,epocas[etapa]):

            for patron in patrones:

                # Hallar la neurona ganadora
                vec_distancias = np.sum((vec_neuronas - patron)**2, axis=1)
                ind_neurona_ganadora = np.argmin(vec_distancias)

                # Buscar las neuronas vecinas
                cord_neurona_ganadora = coordenadas[ind_neurona_ganadora]

                if cuadrado:
                    # Distancia Chebyshev
                    distancias_topologicas = np.max(np.abs(coordenadas - cord_neurona_ganadora), axis=1)
                else:
                    # Distancia Manhattan
                    distancias_topologicas = np.sum(np.abs(coordenadas - cord_neurona_ganadora), axis=1)


                # Mascara para identificar las neuronas vecinas
                mascara_vecindad = distancias_topologicas <= cant_vecinos_etapa

                vec_neuronas[mascara_vecindad] += valor_eta_etapa * (patron - vec_neuronas[mascara_vecindad])

    return vec_neuronas

def activar_neurona(patron, vec_neuronas, ind_neurona, ind_neurona_origen, neuronas_frontera_sur, neuronas_frontera_norte, es_norte_sur, vecindad, eta, filas, columnas):
    if vecindad > 0:

        vecindad -= 1
        cant_neuronas = filas * columnas

        if es_norte_sur:
            if ind_neurona not in neuronas_frontera_sur:
                ind_neurona_s = ind_neurona - 1
                if (ind_neurona_s >= 0 and ind_neurona_s < cant_neuronas and ind_neurona_s!=ind_neurona_origen):
                    activar_neurona(patron, vec_neuronas, ind_neurona_s, ind_neurona, neuronas_frontera_sur, neuronas_frontera_norte, True, vecindad, eta, filas, columnas)

            if ind_neurona not in neuronas_frontera_norte:
                ind_neurona_n = ind_neurona + 1
                if (ind_neurona_n >= 0 and ind_neurona_n < cant_neuronas and ind_neurona_n!=ind_neurona_origen):
                    activar_neurona(patron, vec_neuronas, ind_neurona_n, ind_neurona, neuronas_frontera_sur, neuronas_frontera_norte, True, vecindad, eta, filas, columnas)
        
        ind_neurona_e = ind_neurona - filas
        if (ind_neurona_e >= 0 and ind_neurona_e < cant_neuronas and ind_neurona_e!=ind_neurona_origen):
            activar_neurona(patron, vec_neuronas, ind_neurona_e, ind_neurona, neuronas_frontera_sur, neuronas_frontera_norte, False, vecindad, eta, filas, columnas)

        ind_neurona_o = ind_neurona + filas
        if (ind_neurona_o >= 0 and ind_neurona_o < cant_neuronas and ind_neurona_o!=ind_neurona_origen):
            activar_neurona(patron, vec_neuronas, ind_neurona_o, ind_neurona, neuronas_frontera_sur, neuronas_frontera_norte, False, vecindad, eta, filas, columnas)
        
    vec_neuronas[ind_neurona] += eta * (patron - vec_neuronas[ind_neurona])
    return vec_neuronas

def som_recursivo(patrones, dimension_mapa, epocas=[500, 1000, 3000], eta=0.9, vecindad=1, cuadrado=False, graf=False):

    filas, columnas = dimension_mapa
    cant_neuronas = filas * columnas
    cant_entradas = len(patrones[0,:])
    cant_etapas = len(epocas)

    rng = np.random.default_rng()

    # Vector con los pesos de cada neurona numerada desde 0 hasta cant_neuronas
    vec_neuronas = rng.random((cant_neuronas,cant_entradas))

    neuronas_frontera_sur = set(np.arange(0, cant_neuronas, filas))
    neuronas_frontera_norte = set(np.arange(filas - 1, cant_neuronas, filas))

    m = (1-vecindad)/(cant_etapas-1) if cant_etapas > 1 else 0
    cant_vecinos = lambda x: int(m*x + vecindad) 

    a = eta
    b = np.log(0.1/eta) / (cant_etapas-1) if cant_etapas > 1 else 0
    valor_eta = lambda x: a*np.exp(b*x)

    for etapa in range(0,cant_etapas):

        cant_vecinos_etapa = cant_vecinos(etapa)
        valor_eta_etapa = valor_eta(etapa)
        for epoca in range(0,epocas[etapa]):

            for patron in patrones:

                # Hallar la neurona ganadora
                vec_distancias = np.sum((vec_neuronas - patron)**2, axis=1)
                ind_neurona_ganadora = np.argmin(vec_distancias)

                vec_neuronas = activar_neurona(patron, vec_neuronas, ind_neurona_ganadora, ind_neurona_ganadora, neuronas_frontera_sur, neuronas_frontera_norte, True, cant_vecinos_etapa, valor_eta_etapa, filas, columnas)

    return vec_neuronas


# patrones = np.loadtxt('Guia4/circulo.csv', delimiter=',', skiprows=0)
patrones = np.loadtxt('Guia4/te.csv', delimiter=',', skiprows=0)
# patrones_trn = np.loadtxt('Guia2/iris81_trn.csv', delimiter=',', skiprows=0)
# patrones = patrones_trn[:,0:4]
dimension_mapa = (10,20)

inicio = tm.perf_counter()
vec_neuronas_1 = som_mascara(patrones, dimension_mapa, epocas=[500, 1000, 3000], eta=0.9, vecindad=5, cuadrado=False, graf=False)
fin = tm.perf_counter()
tiempo_total = fin - inicio
print(f"El entrenamiento del SOM demoró: {tiempo_total:.4f} segundos")

# inicio = tm.perf_counter()
# som(patrones, dimension_mapa, epocas=[100, 100, 100], eta=0.9, vecindad=1, cuadrado=True, graf=False)
# fin = tm.perf_counter()
# tiempo_total = fin - inicio
# print(f"El entrenamiento del SOM demoró: {tiempo_total:.4f} segundos")


inicio = tm.perf_counter()
vec_neuronas = som_recursivo(patrones, dimension_mapa, epocas=[500, 1000, 3000], eta=0.9, vecindad=5, cuadrado=False, graf=False)
fin = tm.perf_counter()
tiempo_total = fin - inicio
print(f"El entrenamiento del SOM demoró: {tiempo_total:.4f} segundos")

inicio = tm.perf_counter()
matriz_neuronas_2 = sm.som(patrones, dimension_mapa, epocas=[500, 1000, 3000], eta=0.9, vecindad=5, cuadrado=False, graf=False)
fin = tm.perf_counter()
tiempo_total = fin - inicio
print(f"El entrenamiento del SOM demoró: {tiempo_total:.4f} segundos")

plt.figure(0, figsize=(8,6))
plt.scatter(vec_neuronas[:,0],vec_neuronas[:,1],color='blue',label="Pesos de las neuronas 1")
plt.scatter(vec_neuronas_1[:,0],vec_neuronas_1[:,1],color='green',label="Pesos de las neuronas 2")
plt.scatter(matriz_neuronas_2[:,:,0],matriz_neuronas_2[:,:,1],color='red',label="Pesos de las neuronas 3")
plt.title(f"Incializacion aleatoria de los pesos")
plt.legend()
plt.show()