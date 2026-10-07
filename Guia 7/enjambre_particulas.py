import numpy as np


def enjambre_particulas(cant_particulas,f,entorno,condiciones_iniciales,it_max,tol,aceleracion):
    # f: es la funcion a miniminzar
    #entorno: es el indice de vecindad a tomar
    #condiciones_iniciales: matriz de cant_dimensiones X 2 (la primer columna tiene el minimo y la segunda el maximo)
    #aceleracion: vector con dos comoponetes: la aceleracion para la direccion de la mejor local y el otro de la mejor vecina

    ####################### Incicializacion ##################################
    dimension= len(condiciones_iniciales)
    print(dimension)
    enjambre= np.zeros((cant_particulas,dimension))
    for i in range(cant_particulas):
        for j in range(dimension):
            enjambre[i,j] = np.random.rand()*(condiciones_iniciales[j,1] - condiciones_iniciales[j,0]) + condiciones_iniciales[j,0]

    velocidades=np.zeros((cant_particulas,dimension)) #incializamos la velocidad en cero
    print(enjambre)
    print(velocidades)


    ################### iterar ###############################################

    #inicializamos los mejores con las pociciones iniciales
    mejores_pociciones_locales=enjambre
    #obtnemos los resultados inciales
    resultados_enjambre= np.zeros(cant_particulas)
    for i in range(cant_particulas):
        
        resultados_enjambre[i]= f(enjambre[i,:])
    
    mejores_resultados_locales = resultados_enjambre
    mejores_indices_entorno= np.zeros((cant_particulas),dtype=int)#para que contenga enteros y no flotantes (si no no andan los indices)
    
    it =0
    minimo=float('inf')
    
    mejor_particula=np.zeros(dimension)
    while(it<it_max and minimo>tol):
        #buscamos los mejores entre cada particula
        for i in range(cant_particulas):

            #obtenemos la mejor pos personal
            if(resultados_enjambre[i] < mejores_resultados_locales[i]):#actualizamos la mejor salida y la mejor poscicion personal de cada particula
                mejores_resultados_locales[i]=resultados_enjambre[i]
                mejores_pociciones_locales[i,:]=enjambre[i,:]


            #obtenemos el mejor local de cada vecindad
            mejor_entorno=resultados_enjambre[i]
            pos_mejor_entorno=i
            #buscamos el mejor a la derecha
            for j in range(1,entorno+1):
                indice_abajo = int((i + j)%cant_particulas)
                indice_arriba = int(i - j)

                if(resultados_enjambre[indice_abajo] < mejor_entorno):
                    mejor_entorno = resultados_enjambre[indice_abajo]
                    pos_mejor_entorno= indice_abajo

                if(resultados_enjambre[indice_arriba] < mejor_entorno):
                    mejor_entorno = resultados_enjambre[indice_arriba]
                    pos_mejor_entorno= indice_arriba

            mejores_indices_entorno[i]= pos_mejor_entorno

        # actualizamos la pocicion y velocidad de cada particula
        
        for i in range(cant_particulas):
            r= np.random.rand(2,dimension) #creamos la variable aleatoria para cada particula
            #print(r)
            delta_velocidad = aceleracion[0]*r[0,:]*(mejores_pociciones_locales[i,:] - enjambre[i,:]) + aceleracion[1]*r[1,:]*(enjambre[mejores_indices_entorno[i],:] - enjambre[i,:])
            #print(f"velocidad de la particula {i}: {delta_velocidad}")
            velocidades[i] = delta_velocidad
            #print(velocidades[i])

            enjambre[i,:] = enjambre[i,:] + velocidades[i,:]

            #controlamos que no exceda el limite de las condiciones
            for j in range(dimension):
                if(enjambre[i,j]<condiciones_iniciales[j,0]):
                    enjambre[i,j]=condiciones_iniciales[j,0]
                else:
                    if(enjambre[i,j]>condiciones_iniciales[j,1]):
                        enjambre[i,j]= condiciones_iniciales[j,1]

            #actualizamos el resultado de esa particula
            resultados_enjambre[i] = f(enjambre[i,:])

        pos_menor_resultado=np.argmin(resultados_enjambre)

        if(resultados_enjambre[pos_menor_resultado]< minimo):
            minimo = resultados_enjambre[pos_menor_resultado] #actualizamos el minimo para ver si corta
            mejor_particula=enjambre[pos_menor_resultado,:].copy()# actualiazamos la mejor particula encontrada (pongo el copy porque si no se modifica por referencia... medio raro)
            # print(minimo)
            # print(f(mejor_particula))
            # print(mejor_particula)
        
        print(f"minimo_actual= {resultados_enjambre[pos_menor_resultado]}|| minimo global= {minimo} || iteracion= {it} ")
        it += 1 #incrementamos las iteraciones
    print(mejor_particula)
    return mejor_particula

            

            
                
                    
            