import numpy as np
import matplotlib.pyplot as plt

import time as tm
# inicio = tm.perf_counter()
# fin = tm.perf_counter()
# tiempo_total = fin - inicio
# print(f"El entrenamiento del SOM demoró: {tiempo_total:.4f} segundos")

def algoritmo_genetico(individuos, funcion_decodificacion, funcion_aptitud, mecanismo_seleccion, f_operadores_variacion, f_reproduccion, f_reemplazo, aptitud_requerida, itmax, cant_variables, graf):
    # individuos: [b,p] cantidad de bits para el genotipo y cantidad de individuos en la poblacion
    # funcion_aptitud: promedio de error, estadisticas, correlaciones, distancias, etc.
    # mecanismo_seleccion: Ruleta, Ventanas o Competencias
    # operadores_variacion: Mutacion
    # reproduccion: Cruzas Simples
    # reemplazo: Total, con Brecha generacional o Elitismo

    rng = np.random.default_rng()

    # inicializar(Población)
    cant_bits = individuos[0]
    cant_individuos = individuos[1]

    poblacion = rng.integers(0, 2, size=(cant_individuos, cant_bits))
    # print(f'Poblacion = {poblacion}')

    # MejorAptitud ← Evaluar(Población)
    fenotipos = np.zeros((cant_individuos,cant_variables))
    for i in range(cant_individuos):
        fenotipos[i, :] = funcion_decodificacion(poblacion[i,:])
    # print(f'Fenotipos = {fenotipos}')

    mejores_aptitudes = []

    aptitudes = funcion_aptitud(fenotipos)
    indice_mejor_apto = np.argmax(aptitudes)
    if graf:
        mejores_aptitudes.append(aptitudes[indice_mejor_apto])
        

    iteracion = 1
    # mientras MejorAptitud < AptitudRequerida
    while iteracion < itmax:

    #   Progenitores ← SelecciónNatural(Población)
        ind_progenitores = mecanismo_seleccion(aptitudes)
        # print(f"Indices Progenitores = {ind_progenitores}")

    #   Población ← ReproducciónVariación(Progenitores)
        progenitores_adaptados = f_operadores_variacion(poblacion[ind_progenitores])
        # print(f'Progenitores = {progenitores_adaptados}')

        hijos = f_reproduccion(progenitores_adaptados)
        # print(f'Hijos = {hijos}')

        poblacion = f_reemplazo(poblacion, hijos, aptitudes)
        # print(f'Nueva Poblacion = {poblacion}')

    #   MejorAptitud ← Evaluar(Población)
        cant_individuos = len(poblacion[:,0])
        fenotipos = np.zeros((cant_individuos,cant_variables))
        for i in range(cant_individuos):
            fenotipos[i, :] = funcion_decodificacion(poblacion[i,:])
    
        aptitudes = funcion_aptitud(fenotipos)
        indice_mejor_apto = np.argmax(aptitudes)
        if graf:
            mejores_aptitudes.append(aptitudes[indice_mejor_apto])
        print(f'Iteracion = {iteracion} | Fitness = {aptitudes[indice_mejor_apto]:.4f} | Cant. Individuos = {cant_individuos}')
        # print(f'Iteracion = {iteracion} | Mejor Genotipo = {poblacion[indice_mejor_apto,:]} | Fitness = {aptitudes[indice_mejor_apto]} | Cant. Individuos = {cant_individuos}')

        if (aptitudes[indice_mejor_apto] > aptitud_requerida): 
            break

        iteracion += 1
    # fin

    if graf:
        plt.figure(figsize=(8, 5))
        plt.plot(mejores_aptitudes, color='teal', linewidth=2)
        plt.title('Evolución de la Mejor Aptitud (Fitness)')
        plt.xlabel('Iteración / Generación')
        plt.ylabel('Aptitud (Fitness)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return poblacion[indice_mejor_apto]

def mecanismo_ventanas(aptitudes):
    cant_individuos = len(aptitudes)
    # Ordenar de mayor a menor (los mejores al principio)
    indices_individuos = np.argsort(aptitudes)[::-1]
    progenitores = []

    # Generamos la misma cantidad de progenitores que la población actual
    for _ in range(cant_individuos):
        # Seleccionamos aleatoriamente, pero sesgado hacia los mejores 
        # (ej. elegimos aleatoriamente entre la mitad superior de la población)
        limite_ventana = max(2, cant_individuos // 2) 
        ind_progenitor = np.random.randint(0, limite_ventana)
        progenitores.append(indices_individuos[ind_progenitor])
    
    return progenitores

def mutacion(poblacion):
    rng = np.random.default_rng()
    
    # poblacion.shape[0] te da la cantidad de filas (individuos)
    for i in range(poblacion.shape[0]):
        val_prob = rng.integers(0, 10)
        
        if val_prob < 3: # 30% de individuos que mutan
            # poblacion.shape[1] te da la cantidad de columnas (genes/bits)
            ind_random = rng.integers(0, poblacion.shape[1])
            poblacion[i, ind_random] ^= 1

    return poblacion

def cruzas_simples(poblacion): 
    rng = np.random.default_rng()
    hijos = []
    
    for i in range(0, len(poblacion)-1, 2):
        hijo1 = poblacion[i,:].copy()
        hijo2 = poblacion[i+1,:].copy()

        # Elegir un punto de cruza aleatorio para ESTE par de padres
        # Entre 1 y la cantidad de bits - 1
        cruza = rng.integers(1, poblacion.shape[1])

        hijo1[0:cruza] = poblacion[i+1, 0:cruza]
        hijo2[0:cruza] = poblacion[i, 0:cruza]

        hijos.append(hijo1)
        hijos.append(hijo2)

    return np.array(hijos)

def elitismo(poblacion, hijos, aptitudes):
    ind_mejor = np.argmax(aptitudes)
    individuo_elite = poblacion[ind_mejor, :].copy()
    
    nueva_poblacion = hijos.copy()
    # Reemplazamos el último hijo (o cualquier otro) por el élite
    nueva_poblacion[-1] = individuo_elite
    
    return nueva_poblacion
