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