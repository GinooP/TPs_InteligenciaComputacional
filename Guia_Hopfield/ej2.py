import numpy as np
import red_hopfield as rh
import matplotlib.pyplot as plt


patrones_digitos = {
    0: np.array([
        [-1, -1,  1,  1, -1],
        [-1,  1, -1, -1,  1],
        [-1,  1, -1, -1,  1],
        [-1,  1, -1, -1,  1],
        [-1,  1, -1, -1,  1],
        [-1,  1, -1, -1,  1],
        [-1, -1,  1,  1, -1]
    ]).flatten(),

    1: np.array([
        [-1, -1,  1, -1, -1],
        [-1,  1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1,  1,  1,  1, -1]
    ]).flatten(),

    2: np.array([
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [-1,  1,  1,  1,  1]
    ]).flatten(),

    3: np.array([
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1],
        [-1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1]
    ]).flatten(),

    4: np.array([
        [-1, -1, -1,  1, -1],
        [-1, -1,  1,  1, -1],
        [-1,  1, -1,  1, -1],
        [ 1, -1, -1,  1, -1],
        [ 1,  1,  1,  1,  1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1]
    ]).flatten(),

    5: np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1]
    ]).flatten(),

    6: np.array([
        [-1, -1,  1,  1, -1],
        [-1,  1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1]
    ]).flatten(),

    7: np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1]
    ]).flatten(),

    8: np.array([
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1]
    ]).flatten(),

    9: np.array([
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1,  1, -1],
        [-1,  1,  1, -1, -1]
    ]).flatten()
}

matriz_patrones =  [patrones_digitos[0],
                    patrones_digitos[1],
                    patrones_digitos[2],
                    patrones_digitos[3],
                    patrones_digitos[4],
                    patrones_digitos[5],
                    patrones_digitos[6],
                    patrones_digitos[7],
                    patrones_digitos[8],
                    patrones_digitos[9]] 

dimension = [7,5]
nros_actuales = []
while True:
    n = int(input('Ingrese el numero a graficar (-1 para finalizar):'))

    if (n == -1):
        break
        
    if (n not in nros_actuales):

        nros_actuales = [n]

        relacion = np.array(matriz_patrones) @ patrones_digitos[n]
        relacion = np.abs(relacion) / np.max(relacion)

        patrones = [patrones_digitos[n]]
        for i in range(0,3):
            nro = np.argmin(relacion)
            patrones.append(patrones_digitos[nro])
            relacion[nro] = 1
            nros_actuales.append(nro)

        W = rh.red_hopfield(np.array(patrones).copy())
        
    patron = patrones_digitos[n].copy()
    y, historial, epoca = rh.recuperar_rh(patron, W)
    
    imagen = y.reshape((dimension[0], dimension[1])) # convertir el vector 1D a matriz 2D

    # graficar los pixeles
    plt.imshow(imagen, cmap='gray_r', vmin=-1, vmax=1)
    plt.title("Patrón Recuperado")
    plt.axis('off') # Oculta las marcas de los ejes para que se vea limpio
    plt.show()
    plt.close()