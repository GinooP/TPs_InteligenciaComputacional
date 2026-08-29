import numpy as np
import red_hopfield as rh
import matplotlib.pyplot as plt
import time

patrones = np.array([
    [-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1],
    [1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,1],
    [1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1]
])
W = rh.red_hopfield(patrones)

patron = np.array([1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,1])

patron_ruidoso = rh.agregar_ruido(patron, 0.1)

y, historial, epoca = rh.recuperar_rh(patron_ruidoso,W)
print(y)

dimension = int(np.sqrt(len(y))) # tamaño de la cuadricula

imagen = y.reshape((dimension, dimension)) # convertir el vector 1D a matriz 2D


# for paso, estado in enumerate(historial):
#     imagen = estado.reshape((dimension, dimension))
#     plt.imshow(imagen, cmap='gray_r', vmin=-1, vmax=1)
#     plt.title(f"Iteración {paso} para 10% de ruido")
#     plt.axis('off')
#     plt.pause(0.5) # Pausa de 0.1 segundos entre cuadros
#     plt.clf() # Limpia la figura para el siguiente paso


patron_ruidoso = rh.agregar_ruido(patron, 0.2)

y, historial, epoca = rh.recuperar_rh(patron_ruidoso,W)
print(y)

dimension = int(np.sqrt(len(y))) # tamaño de la cuadricula

imagen = y.reshape((dimension, dimension)) # convertir el vector 1D a matriz 2D

# for paso, estado in enumerate(historial):
#     imagen = estado.reshape((dimension, dimension))
#     plt.imshow(imagen, cmap='gray_r', vmin=-1, vmax=1)
#     plt.title(f"Iteración {paso} para 20% de ruido")
#     plt.axis('off')
#     plt.pause(0.5) # Pausa de 0.1 segundos entre cuadros
#     plt.clf() # Limpia la figura para el siguiente paso

# 20% DE RUIDO
patron_ruidoso = rh.agregar_ruido(patron, 0.5)

y, historial, epoca = rh.recuperar_rh(patron_ruidoso,W)
print(y)

dimension = int(np.sqrt(len(y))) # tamaño de la cuadricula

imagen = y.reshape((dimension, dimension)) # convertir el vector 1D a matriz 2D

# for paso, estado in enumerate(historial):
#     imagen = estado.reshape((dimension, dimension))
#     plt.imshow(imagen, cmap='gray_r', vmin=-1, vmax=1)
#     plt.title(f"Iteración {paso} para 50% de ruido")
#     plt.axis('off')
#     plt.pause(0.5) # Pausa de 0.1 segundos entre cuadros
#     plt.clf() # Limpia la figura para el siguiente paso


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
porcentaje = float(input('Porcentaje de ruido: '))
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
    patron_ruidoso = rh.agregar_ruido(patron, porcentaje)
    y, historial, epoca = rh.recuperar_rh(patron_ruidoso, W)

    imagen = y.reshape((dimension[0], dimension[1])) # convertir el vector 1D a matriz 2D

    for paso, estado in enumerate(historial):
        imagen = estado.reshape((dimension[0], dimension[1]))
        plt.imshow(imagen, cmap='gray_r', vmin=-1, vmax=1)
        plt.title(f"Iteración {paso} para {porcentaje*100}% de ruido")
        plt.axis('off')
        plt.pause(0.3) # Pausa de 0.3 segundos entre cuadros
        plt.clf() # Limpia la figura para el siguiente paso
