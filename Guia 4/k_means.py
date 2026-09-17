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
    
    lotes_patrones = np.zeros((N,1))
    centroides = np.zeros((nro_grupos, M))

    for i in range(epocas):
        centroides[:,:] = 0
        cant_patrones_centroide = np.zeros((nro_grupos,1))
        # 2. Dividir los patrones en los lotes
        for j in range(N):
            # 2.1. Calcular la distancia a los K
            distancia = np.sum((K - patrones[j,:])**2,axis=1)

            # 2.2. Agrupar los patrones
            ind_lote = np.argmin(distancia)
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
        K = centroides

    return K, lotes_patrones

def calcular_compactitud(k,lotes_patrones,entradas):
    cant_clusters = len(k[:,0])
    cant_patrones = len(entradas[:,0])

    compactidudes=np.zeros((cant_clusters))
    cont_cant_patrones_x_cluster=np.zeros((cant_clusters))
    for i in range(cant_patrones):
        #distancia = sum((k - entradas[i,:])**2)
        distancia= np.linalg.norm(k - entradas[i,:],2)

        ind_cluster=int(lotes_patrones[i])
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