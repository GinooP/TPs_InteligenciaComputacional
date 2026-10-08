import numpy as np
import time as tm


def algoritmo_enjambre_particulas(
       cant_particulas,     # Cantidad de particulas que vamos a inicializar
       limites_dimensiones, # Matriz de 2xN con los valores minimos y maximos para cada dimension
       aceleracion,         # Vector de 2 componentes que contiene la aceleracion c1 y c2
       funcion,             # Funcion a minimizar
       cant_vecinos,        # Cantidad de vecinos por lado que tendremos para cada particula (entorno)
       aptitud_requerida,   # Criterio de finalizacion para la mejor posicion
       maximo_iteraciones   # Cantidad maxima de iteraciones sin converger
):
    tini = tm.perf_counter()
    rng = np.random.default_rng()
    cant_dimensiones = limites_dimensiones.shape[1]
    c1,c2 = aceleracion
    salida = np.zeros(cant_dimensiones)
    convergio = False

    # 1. Inicializar las particulas en posiciones aleatorias
    X = np.zeros((cant_particulas,cant_dimensiones))
    for i in range(cant_dimensiones):
        X[:,i] = rng.uniform(limites_dimensiones[0,i], limites_dimensiones[1,i], cant_particulas)
    
    Y = X.copy()    # Mejor posicion de la particula inicial
    Y_aptitudes = funcion(Y)

    Yentorno = X.copy() # Mejor posicion de cada entorno inicial
    Yentorno_aptitudes = funcion(Yentorno)

    # 2. Inicializar las mejores posiciones de los entornos
    for k in range(cant_particulas):
        indices_entorno = [k]
        if cant_vecinos != 0:
            indices_entorno = list(range(k-cant_vecinos,k+cant_vecinos+1))

        aptitud_mejor_posicion_entorno = Yentorno_aptitudes[k]
        for i in indices_entorno:

            indice = i%cant_particulas
            aptitud_particula = Y_aptitudes[indice]

            if aptitud_particula < aptitud_mejor_posicion_entorno:
                Yentorno[k,:] = Y[indice,:]
                aptitud_mejor_posicion_entorno = Y_aptitudes[indice]

    X_aptitudes = funcion(X)
    Y_aptitudes = funcion(Y)
    Yentorno_aptitudes = funcion(Yentorno)

    # 3. Inicializar las velocidades iniciales en 0
    V = np.zeros((cant_particulas,cant_dimensiones))

    # 4. Mientras que no se cumpla el criterio o llegue al maximo de iteraciones
    for it in range(maximo_iteraciones):

        # 4.1. Recorrer las particulas:
        for k in range(cant_particulas):

            # 4.1.1. Actualizar la mejor posicion de la particula de ser necesario
            aptitud_particula = X_aptitudes[k]
            aptitud_mejor_posicion = Y_aptitudes[k]
            if (aptitud_particula < aptitud_mejor_posicion):
                Y[k,:] = X[k,:]
                Y_aptitudes[k] = X_aptitudes[k]
            
            # 4.1.2. Actualizar la mejor posicion del entorno de la particula de ser necesario
            indices_entorno = [k]
            if cant_vecinos != 0:
                indices_entorno = list(range(k-cant_vecinos,k+cant_vecinos+1))
    
            aptitud_mejor_posicion_entorno = Yentorno_aptitudes[k]
            for i in indices_entorno:
    
                indice = i%cant_particulas
                aptitud_particula = Y_aptitudes[indice]
    
                if aptitud_particula < aptitud_mejor_posicion_entorno:
                    Yentorno[k,:] = Y[indice,:]
                    Yentorno_aptitudes[k] = Y_aptitudes[indice]
            
            
        # 4.2. Recorrer las particulas:
        for k in range(cant_particulas):

            r1 = rng.random(cant_dimensiones)
            r2 = rng.random(cant_dimensiones)

            # 4.2.1. Actualizar la velocidad de la particula
            w = 0.7 # Inercia
            V[k,:] = w*V[k,:] + c1*r1*(Y[k,:] - X[k,:]) + c2*r2*(Yentorno[k,:] - X[k,:])

            # 4.2.2. Actualizar la posicion de la particula
            X[k,:] += V[k,:]
            X[k,:] = np.clip(X[k, :], limites_dimensiones[0, :], limites_dimensiones[1, :])

        X_aptitudes = funcion(X)


        # 4.3. Chequear el cumplimiento del criterio
        for k in range(cant_particulas):
            aptitud_particula = Yentorno_aptitudes[k]
            if (aptitud_particula < aptitud_requerida):
                salida = Yentorno[k,:]
                print(f'El algoritmo convergio al valor {salida} en la iteracion {it}')
                convergio = True
                break
        
        if convergio:
            break

        salida = Yentorno[np.argmin(Yentorno_aptitudes),:]

        print(f'Iteracion = {it} | Mejor Global = {salida}')

    tfin = tm.perf_counter()
    ttotal = tfin - tini
    print(f"\nEl algoritmo de Enjambre de Particulas demoró: {ttotal:.4f} segundos \n")

    # 4. Devolver la mejor particula encontrada

    return salida, ttotal



