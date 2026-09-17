import numpy as np
import matplotlib.pyplot as plt

def buscar_indice_activacion(matriz_neuronas,patron):

    distancias = np.sum((patron - matriz_neuronas)**2, axis=2)
    pos_act = np.unravel_index(np.argmin(distancias), distancias.shape)
    # #busca la menor distancia
    # #primero busca en la primera fila
    # #print(f"matriz_neuironas= {matriz_neuronas[0,0]}")
    # #print(f"patron: {patron}")
    # menor= sum((patron - matriz_neuronas[0,0])**2)
    # #print(f"este es el menor: {menor}")
    # pos_act=(0,0)
    # for j  in range(1,dim_matriz_neuronas[1]):
    #     dist = sum((patron - matriz_neuronas[0,j])**2)#distancia euclidea (sin la raiz cuadrada)

    #     if(dist < menor):
    #         menor=dist
    #         pos_act=(0,j)

    # #luego busca en el resto de la matriz
    # for i in range(dim_matriz_neuronas[0]):
    #     for j in range(dim_matriz_neuronas[1]):
    #         dist = sum((patron - matriz_neuronas[i,j])**2)#distancia euclidea (sin la raiz cuadrada)

    #         if(dist < menor):
    #             menor=dist
    #             pos_act=(i,j)

    return pos_act

def som(patrones,dim_matriz_neuronas,epocas,eta,vecindad,cuadrado,graf,tol=1e-5):
    #cuadrado=indica si la vencidad tiene forma cuadrada o de rombo
    #vecindad = determina hasta cuantos vecinos tomamos (con forma de rombo o cuadrado dependiendo del bool
    #dim_matriz_neuronas: dimensiones de la matriz de neuronas (filas,columnas)
    #epocas podria ser un vector con la camtidad de epocas que queramos en cada etapa

    ###################### inicilaizacion de los pesos ###################################
    N=len(patrones[0,:])#obtnemos la cantidad de columanas para determinar el tamaño del vector de pesos
    cant_patrones=len(patrones[:,1])#obtenemos la cantidad de patrones

    rng = np.random.default_rng()
    matriz_neuronas= rng.random(size=(dim_matriz_neuronas[0],dim_matriz_neuronas[1],N)) - 0.5

    ##################################
    m=(1 - vecindad)/epocas[1]#pendiente de la recta
    h= vecindad #termino independiente de la recta
    dec_lineal_vecindad= lambda x: np.trunc(m*x + h) # decremento lineal de la vecindad desde su valor inicial hasta 1
                                            
    b=eta
    a=np.log(0.1/b)/epocas[1]
    dec_expo_eta = lambda x: b*np.exp(a*x)# decremento exponencial del coeficiente de aprendisaje (eta) desde su valor inicial hasta 0.1

    # X=np.linspace(0,epocas[1])
    # Y =dec_expo_eta(X)
    # plt.figure(1)
    # plt.plot(X,Y)
    # plt.plot(0,eta)
    # plt.plot(epocas[1],0.1)
    # plt.grid(visible=True)
    # plt.show()

    # #graficamos las neuronas con las pociciones al azar}
    # plt.ion()
    # plt.figure(0,figsize=(8,6))
    # plt.scatter(matriz_neuronas[:,:,0],matriz_neuronas[:,:,1],color='blue',label="Pesos de las neuronas")
    # plt.scatter(patrones[:,0],patrones[:,1],color='red',alpha=0.5,label="Patrones")
    # plt.title(f"Incializacion aleatoria de los pesos")
    # plt.legend()
    
    etapa=0
    while(etapa<3):
        
        for n in range(epocas[etapa]):
            print(f"epocas={n}")
            matriz_neuronas_ini =  matriz_neuronas.copy() #matriz para comprobar si los desplazamientos son grandes
            for k in range(cant_patrones):#recorremos los patrones
                #buscamos la neurona que debe activarse
                pos_act=buscar_indice_activacion(matriz_neuronas,patrones[k]) # funciona pero lo podria cambiar por una operacion vectorial ahora que solucione las dimensiones

                ############## actualizamos los pesos ###########################
                if(vecindad!=0):
                    #obtenemos desplazamientos hacia la izquierda, derecha ,arriba y abajo para no salirnos del margen
                    #desp_arriba
                    if(pos_act[0] - vecindad < 0):
                        desp_arriba=pos_act[0] #si i_act=0 no me puedo desplazar para arriba. si i_act=1 solo me puedo desplazar un lugar para arriba y así
                    else:
                        desp_arriba=vecindad

                    #desp_abajo
                    if(pos_act[0] + vecindad >= dim_matriz_neuronas[0]):
                        desp_abajo=dim_matriz_neuronas[0]-1 - pos_act[0]
                    else:
                        desp_abajo=vecindad

                    #desp_izq
                    if(pos_act[1] - vecindad < 0):
                        desp_izq=pos_act[1]#
                    else:
                        desp_izq=vecindad

                    #desp_derecha
                    if(pos_act[1] + vecindad >= dim_matriz_neuronas[1]):
                        desp_der= dim_matriz_neuronas[1]-1 - pos_act[1]
                    else:
                        desp_der= vecindad
                    
                    if(cuadrado==True):
                        #con esto genero un cuadrado 

                        delta_pesos=eta*(patrones[k] - matriz_neuronas[pos_act[0]-desp_arriba:pos_act[0]+desp_abajo, pos_act[1]-desp_izq:pos_act[1]+desp_der])
                        #print(f"\n {delta_pesos}\n")
                        matriz_neuronas[pos_act[0]-desp_arriba:pos_act[0]+desp_abajo, pos_act[1]-desp_izq:pos_act[1]+desp_der] += delta_pesos
                    else:
                        #con esto formamos un rombo

                        #esto no funciona
                        # filas, columnas,profudidad = np.indices(matriz_neuronas.shape)
                        # matriz_neuronas[abs(filas - pos_act[0]) + abs(columnas - pos_act[1]) <= vecindad] += eta*(patrones[k] - matriz_neuronas[abs(filas - pos_act[0]) + abs(columnas - pos_act[1]) <= vecindad])

                        #recorremos un rango desde la esquina superior izquierda hasta la esquina inferior derecha
                        # de un rango reducido de celdas para determinar si estan dentro de la vecindad para actualizar sus pesos
                        i_inicio= pos_act[0]-desp_arriba
                        i_fin=pos_act[0] + desp_abajo

                        j_inicio= pos_act[1] - desp_izq
                        j_final= pos_act[1] + desp_der
                        for i in range(i_inicio,i_fin+1):#le sumamos uno por que el rango no concidera el ultimo valor (si no llega hasta i_fin-1)
                            for j in range(j_inicio,j_final+1):
                                #si la distancia manhatan es menor a la vecindad concideramos que esta dentro de esta
                                #y actualizamos sus pesos
                                if(abs(i - pos_act[0]) + abs(j - pos_act[1]) <= vecindad):
                                    matriz_neuronas[i,j]+= eta*(patrones[k] - matriz_neuronas[i,j])

                else:
                    #actualizamos la neurona individualmente
                    matriz_neuronas[pos_act[0],pos_act[1]] += eta*(patrones[k] - matriz_neuronas[pos_act[0],pos_act[1]])

            if(etapa==1):
                vecindad = int(dec_lineal_vecindad(n)) #decrece con las epocas hasta 1
                eta = dec_expo_eta(n) #decrece con las epocas hasta 0.1

            #condicion de corte
            if(etapa== 2):
                desp_neuronas = abs(matriz_neuronas_ini - matriz_neuronas)
                if (np.max(desp_neuronas) < tol):#si el desplzamiento maximo es menor a tol cortamos
                    break
        
        etapa += 1#incrementamos la etapa

        if(etapa==2):
            vecindad=0
            eta=0.01

        #graficamos las neuronas (osea sus pesos)
        if graf:
            plt.figure(etapa+1,figsize=(8,6))
            plt.scatter(matriz_neuronas[:,:,0],matriz_neuronas[:,:,1],color='blue',label="Pesos de las neuronas")
            plt.scatter(patrones[:,0],patrones[:,1],color='red',alpha=0.5,label="Patrones")
            plt.title(f"Etapa ={etapa}")
            plt.legend()

            plt.scatter(patrones[:, 0], patrones[:, 1], c='red', label='Patrones', alpha=0.5)

            # 2. Dibujamos las líneas de la malla del SOM
            filas, columnas, _ = matriz_neuronas.shape

            # Dibujar conexiones horizontales (a lo largo de cada fila)
            for i in range(filas):
                # Toma todas las columnas de la fila 'i', coordenada X (índice 0) y Y (índice 1)
                plt.plot(matriz_neuronas[i, :, 0], matriz_neuronas[i, :, 1], color='blue', alpha=0.4)

            # Dibujar conexiones verticales (a lo largo de cada columna)
            for j in range(columnas):
                # Toma todas las filas de la columna 'j', coordenada X (índice 0) y Y (índice 1)
                plt.plot(matriz_neuronas[:, j, 0], matriz_neuronas[:, j, 1], color='blue', alpha=0.4)

            # 3. Finalmente, graficamos los puntos azules (los pesos/neuronas) por encima de las líneas
            # Aplanamos la matriz temporalmente a (81, 2) solo para el scatter
            neuronas_planas = matriz_neuronas.reshape(-1, 2) 
            plt.scatter(neuronas_planas[:, 0], neuronas_planas[:, 1], c='blue', label='Pesos de las neuronas', zorder=5)

            plt.ioff()
            plt.show()
            
    return matriz_neuronas

def clasificar(matriz_neuronas,entradas,salidas):
    filas = len(matriz_neuronas[:,0])
    columnas = len(matriz_neuronas[0,:])
    comp_etiqueta = len(salidas[0,:])

    cant_patrones=len(entradas[:,0])

    clasificacion = np.zeros((filas,columnas,comp_etiqueta))
    for i in range(cant_patrones):
        ind_act = buscar_indice_activacion(matriz_neuronas=matriz_neuronas,patron=entradas[i,:])
        ind_salida=np.argmax(salidas[i,:])
        clasificacion[ind_act[0],ind_act[1],ind_salida] += 1

    print(clasificacion)

    for i in range(filas):
        for j in range(columnas):
            ind_max= np.argmax(clasificacion[i,j,:])
            clasificacion[i,j,:]=-1
            clasificacion[i,j,ind_max]=1

    print(clasificacion)

    return clasificacion
                    
def tst_SOM(matriz_neuronas,entradas,salidas,clasificacion=None):
    cant_salidas=len(salidas[0,:])
    cant_patrones=len(entradas[:,0])

    if(clasificacion is None):
        clasificacion=clasificar(matriz_neuronas,entradas,salidas)

    matriz_contingencia=np.zeros((cant_salidas,cant_salidas))

    for i in range(cant_patrones):
        ind_act=buscar_indice_activacion(matriz_neuronas,patron=entradas[i,:])

        salida_correcta= np.argmax(salidas[i,:])
        salida_predicha= np.argmax(clasificacion[ind_act[0],ind_act[1],:])

        matriz_contingencia[salida_correcta,salida_predicha] += 1

    #print(f"matriz contingencia SOM \n {matriz_contingencia}")
    return matriz_contingencia 
                    

                

                    