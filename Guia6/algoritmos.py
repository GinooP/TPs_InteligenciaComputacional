import numpy as np
import matplotlib.pyplot as plt

import time as tm
# inicio = tm.perf_counter()
# fin = tm.perf_counter()
# tiempo_total = fin - inicio
# print(f"El entrenamiento del SOM demoró: {tiempo_total:.4f} segundos")

def ventanas(aptitudes, k=None, cant_progenitores=None, tasa_de_brecha=None):

    rng = np.random.default_rng()
    cant_individuos = len(aptitudes)

    if cant_progenitores == None:
        cant_progenitores =  cant_individuos // 2   # Elegir la mitad de la cantidad de individuos
    
    if tasa_de_brecha != None: # Va a haber un Reemplazo con Brecha Generacional
        cant_progenitores = int(cant_individuos * tasa_de_brecha)

    indices_de_progenitores = []
    
    ...

def competencia(aptitudes, k, cant_progenitores=None, tasa_de_brecha=None):

    rng = np.random.default_rng()
    cant_individuos = len(aptitudes)

    if cant_progenitores == None:
        cant_progenitores =  cant_individuos // 2   # Elegir la mitad de la cantidad de individuos
    
    if tasa_de_brecha != None: # Va a haber un Reemplazo con Brecha Generacional
        cant_progenitores = int(cant_individuos * tasa_de_brecha)

    indices_de_progenitores = []

    for i in range(cant_progenitores):

        individuos_a_competir = rng.choice(cant_individuos, k, replace=False)

        aptitudes_competidores = aptitudes[individuos_a_competir]
        indice_ganador_local = np.argmax(aptitudes_competidores)

        indice_ganador = individuos_a_competir[indice_ganador_local]

        indices_de_progenitores.append(indice_ganador)

    return np.array(indices_de_progenitores)

def ruleta(aptitudes, k=None, cant_progenitores=None, tasa_de_brecha=None):
    rng = np.random.default_rng()
    cant_individuos = len(aptitudes)

    if cant_progenitores == None:
        cant_progenitores =  cant_individuos // 2   # Elegir la mitad de la cantidad de individuos
    
    if tasa_de_brecha != None: # Va a haber un Reemplazo con Brecha Generacional
        cant_progenitores = int(cant_individuos * tasa_de_brecha)

    indices_de_progenitores = []
    
    ...

def mutacion(poblacion, genes, tasa_de_mutacion):
    rng = np.random.default_rng()
    pob_mutada = poblacion.copy() 

    tm = tasa_de_mutacion[0]     # Probabilidad (float entre 0.0 y 1.0)
    nivel = tasa_de_mutacion[1]  # 'individuo' o 'gen'

    cant_individuos = poblacion.shape[0]
    cant_bits = np.sum(genes)
    
    if nivel == 'individuo':
        # Modificaremos los individuos con una probabilidad de tm
        for i in range(cant_individuos):

            val_aleatorio = rng.random()
            if val_aleatorio < tm:

                j = rng.integers(0, cant_bits)
                pob_mutada[i,j] ^= 1

    elif nivel == 'gen':
        # Modificaremos los genes con una probabilidad de tm
        cant_genes = len(genes)
        for i in range(cant_individuos):

            ind_gen = 0
            for g in range(cant_genes):

                val_aleatorio = rng.random()
                if val_aleatorio < tm:
                    
                    j = rng.integers(ind_gen, ind_gen+genes[g])
                    pob_mutada[i,j] ^= 1

                ind_gen += genes[g]

    return pob_mutada

def cruzas_simples(progenitores, cant_individuos, tasa_de_brecha=None):
    rng = np.random.default_rng()
    cant_progenitores = progenitores.shape[0]
    cant_bits = progenitores.shape[1]

    cant_hijos = cant_individuos
    if tasa_de_brecha is not None:  # Reemplazo con Brecha Generacional
        cant_hijos = cant_individuos - cant_progenitores

    hijos = np.zeros((cant_hijos, cant_bits), dtype=progenitores.dtype)
    for i in range(cant_hijos):

        p1, p2 = rng.choice(cant_progenitores, size=2, replace=False)

        cruza = rng.integers(1, cant_bits)

        hijos[i, :cruza] = progenitores[p1, :cruza]
        hijos[i, cruza:] = progenitores[p2, cruza:]

    return hijos

def total(progenitores_variados, hijos, individuo_mejor_fitness_anterior):
    nueva_poblacion = hijos.copy()
    return nueva_poblacion

def brecha_generacional(progenitores_variados, hijos, individuo_mejor_fitness_anterior):
    nueva_poblacion = np.vstack((progenitores_variados, hijos))
    return nueva_poblacion

def elitismo(progenitores_variados, hijos, individuo_mejor_fitness_anterior):
    nueva_poblacion = hijos.copy()
    nueva_poblacion[-1] = individuo_mejor_fitness_anterior
    return nueva_poblacion


def algoritmo_genetico(
        cant_individuos,                # cantidad de individuos en la poblacion
        genes,                          # vector que contiene la estructura del cromosoma. Ej: [5, 5] -> 5 bits para cada gen del cromosoma
        aptitud_requerida,              # fitness requerido para finalizar el algoritmo y retornar la solución.
        maximo_de_iteraciones=1000,     # cantidad máxima de iteraciones sin converger
        cant_progenitores=None,         # cantidad de padres que van a ser seleccionados en la función de seleccion
        tasa_de_mutacion=None,          # [tm, n]: vector que contiene la probabilidad (tm) de que los individuos muten en la función de variación, definida a nivel de Individuos (n='individuo') o de Genes (n='gen').
        tasa_de_brecha=None,            # es la tasa usada para reemplazo con brecha generacional, que nos dice el porcentaje de progenitores que vamos a tener en la nueva poblacion.
        k=2,
        *,
        f_decodificacion,               # funcion para pasar de genotipo a fenotipo
        f_aptitud,                      # funcion para evaluar el fitness del fenotipo de un individuo. Ej: promedio de error, estadisticas, correlaciones, distancias, etc.
        f_seleccion=competencia,        # funcion para seleccionar los progenitores de la nueva población. Ej: Ruleta, Ventanas o Competencias
        f_variacion=mutacion,           # funcion para variar los genes de los cromosomas que reciba. Ej: Mutacion
        f_reproduccion=cruzas_simples,  # funcion para reproducir los progenitores seleccionados. Ej: Cruzas Simples
        f_reemplazo=brecha_generacional,# funcion para formar la nueva poblacion en base a los progenitores y los hijos formados. Ej: Total, con Brecha generacional o Elitismo
        graf=False                      # booleano para graficar las mejores aptitudes en función de las iteraciones
    ):

    
    inicio = tm.perf_counter()
    mejores_fitness = [] # para graficar

    # 1. Inicializar la poblacion de forma aleatoria
    rng = np.random.default_rng()
    cant_bits_por_individuo = np.sum(genes)
    poblacion = rng.integers(0, 2, size=(cant_individuos, cant_bits_por_individuo))

    # 2. Evaluar el fitness de la poblacion
    fenotipos = f_decodificacion(poblacion, genes)
    aptitudes = f_aptitud(fenotipos)

    # 3. Obtener el individuo de mejor fitness
    indice_individuo_mejor_fitness = np.argmax(aptitudes)
    val_individuo_mejor_fitness = aptitudes[indice_individuo_mejor_fitness]
    if graf:
        mejores_fitness.append(val_individuo_mejor_fitness.copy())

    # 4. Mientras el mejor fitness sea menor que el fitness requerido:
    for it in range(maximo_de_iteraciones):

        # 4.1. Obtener los progenitores
        indices_progenitores = f_seleccion(aptitudes, k, cant_progenitores, tasa_de_brecha)

        # 4.2. Mutar los progenitores
        progenitores_variados = f_variacion(poblacion[indices_progenitores], genes, tasa_de_mutacion)

        # 4.3. Reproducir los progenitores
        hijos = f_reproduccion(progenitores_variados, cant_individuos, tasa_de_brecha)

        # 4.4. Reemplazar la población actual por la nueva generación
        poblacion = f_reemplazo(progenitores_variados, hijos, poblacion[indice_individuo_mejor_fitness])

        # 4.5. Obtener el fitness de la población
        fenotipos = f_decodificacion(poblacion, genes)
        aptitudes = f_aptitud(fenotipos)

        # 4.6. Obtener el individuo de mejor fitness
        indice_individuo_mejor_fitness = np.argmax(aptitudes)
        val_individuo_mejor_fitness = aptitudes[indice_individuo_mejor_fitness]
        
        if graf:
            mejores_fitness.append(val_individuo_mejor_fitness.copy())

        print(f'Fitness = {val_individuo_mejor_fitness:.4f} | Iteracion = {it+1}')
        # print(f'Mejor Genotipo = {poblacion[indice_individuo_mejor_fitness,:]} | Fitness = {val_individuo_mejor_fitness:.4f} | Iteracion = {it+1}')

        if (val_individuo_mejor_fitness > aptitud_requerida): 
            break

    fin = tm.perf_counter()
    tiempo_total = fin - inicio
    print(f"El algoritmo demoró: {tiempo_total:.4f} segundos")

    # Opcional: Graficar los fitness a lo largo del tiempo    
    if graf:
        plt.figure(figsize=(8, 5))
        plt.plot(mejores_fitness, color='teal', linewidth=2)
        plt.title('Evolución de la Mejor Aptitud (Fitness)')
        plt.xlabel('Iteración / Generación')
        plt.ylabel('Aptitud (Fitness)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # 5. Devolver el individuo con mejor fitness
    return poblacion[indice_individuo_mejor_fitness]
