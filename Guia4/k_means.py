import numpy as np

# metodo de clustering k-medias por lotes

def k_means(patrones, nro_grupos, epocas):
    # nro_grupos: cantidad de K que se desean crear
    # patrones: cantidad de datos

    N = len(patrones[:,0])
    M = len(patrones[0,:])

    # 1. Inicializar los K en posiciones aleatorias
    rng = np.random.default_rng()
    indices_random = rng.choice(N, size=nro_grupos, replace=False)
    K = patrones[indices_random,:]

    #incializacion buscando los k mas lejanos entre si
    # indice_random=rng.choice(N,size=1, replace=False)
    # K[0]=patrones[indice_random,:]
    # for i in range(nro_grupos):
    #     for j in range(N):
    #         distancia= sum((K - patrones[j,:])**2)
    #         ind_max=np.argmax(distancia)

        
        
    
    lotes_patrones = np.zeros(N)
    # print(lotes_patrones)
    centroides = np.zeros((nro_grupos, M))

    for i in range(epocas):
        centroides[:,:] = 0
        cant_patrones_centroide = np.zeros((nro_grupos,1))
        # 2. Dividir los patrones en los lotes
        hubo_reasignacion=False#variable bandera que sirve para ver si hubo reasignaciones
        for j in range(N):
            # 2.1. Calcular la distancia a los K
            distancia = np.sum((K - patrones[j,:])**2,axis=1)

            # 2.2. Agrupar los patrones
            ind_lote = np.argmin(distancia)
            if(hubo_reasignacion == False and ind_lote != lotes_patrones[j] ):
                hubo_reasignacion=True #si hubo reasigancion ponemos la bandera en true
            lotes_patrones[j] = ind_lote

            # 3. Calcular los centroides de cada lote
            # 3.1. Acumular valores
            centroides[ind_lote] += patrones[j,:]
            # 3.2. Contar cantidad
            cant_patrones_centroide[ind_lote] += 1 

        for j in range(nro_grupos):
            if cant_patrones_centroide[j] == 0: 
                centroides[j] = K[j]
            else:
                centroides[j] /= cant_patrones_centroide[j]
            
        # 4. Mover el K de cada lote hacia el centroide
        K = centroides.copy()
        if(hubo_reasignacion == False):# si no hubo reasignaciones cortamos
            print(f"convergencia del k-medias en la epoca {i}")
            break

    return K, lotes_patrones

def clasificar_K(K, lotes_patrones, salidas):

    nro_lotes = len(K[:,0])
    nro_patrones = len(lotes_patrones)
    nro_salidas = len(salidas[0,:])

    clasificacion = np.zeros((nro_lotes,nro_salidas))

    for i in range(nro_patrones):
        ind_k = int(lotes_patrones[i])
        ind_c = np.argmax(salidas[i,:])
        # print(ind_k)
        # print(ind_c)
        clasificacion[ind_k, ind_c] += 1

    print(f"resultado del conteo: \n {clasificacion}")

    for i in range(nro_lotes):
        ind_max = np.argmax(clasificacion[i,:])
        clasificacion[i,:] = -1
        clasificacion[i,ind_max] = 1

    return clasificacion

def tst_k_means(K, lotes_patrones, entradas, salidas):
    nro_patrones = len(entradas[:,0])
    nro_salidas = len(salidas[0,:])

    clasificacion = clasificar_K(K, lotes_patrones, salidas)
    print(f"esta es la clasificacion \n {clasificacion}")

    matriz_contingencia = np.zeros((nro_salidas,nro_salidas))
    lotes_patrones = np.zeros((nro_patrones))

    for i in range(nro_patrones):
        
        distancia = np.sum((K - entradas[i,:])**2,axis=1)
        
        # Agrupar los patrones
        ind_lote = np.argmin(distancia)
        lotes_patrones[i] = ind_lote

        # Incrementamos el valor de la matriz
        ind_fila = np.argmax(salidas[i,:])
        ind_col = np.argmax(clasificacion[int(lotes_patrones[i])])
        matriz_contingencia[ind_fila,ind_col] += 1

    return matriz_contingencia

def calcular_compactitud(k,lotes_patrones,entradas):
    cant_clusters = len(k[:,0])
    cant_patrones = len(entradas[:,0])

    compactidudes=np.zeros((cant_clusters))
    cont_cant_patrones_x_cluster=np.zeros((cant_clusters))
    for i in range(cant_patrones):
        #distancia = sum((k - entradas[i,:])**2)
        ind_cluster=int(lotes_patrones[i])
        distancia= np.linalg.norm(k[ind_cluster] - entradas[i,:],2)

        
        print(ind_cluster)
        cont_cant_patrones_x_cluster[ind_cluster] += 1

        compactidudes[ind_cluster] += distancia

    for i in range(cant_clusters):
        if (cont_cant_patrones_x_cluster[i] != 0):
            compactidudes[i]= compactidudes[i]/cont_cant_patrones_x_cluster[i]
        else:
            compactidudes[i]=0

    compactitud_global= sum(compactidudes)/len(compactidudes[:])
    return compactidudes,compactitud_global