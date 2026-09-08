import numpy as np
import red_hopfield as rh
import matplotlib.pyplot as plt

patrones = np.array([
    [-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1],
    [1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,1],
    [1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1]
])
W = rh.red_hopfield(patrones)

patron = np.array([1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1])
y, historial, epoca = rh.recuperar_rh(patron,W)
print(y)

dimension = int(np.sqrt(len(y))) # tamaño de la cuadricula

imagen = y.reshape((dimension, dimension)) # convertir el vector 1D a matriz 2D

# graficar los pixeles
# plt.imshow(imagen, cmap='gray_r', vmin=-1, vmax=1)
# plt.title("Patrón Recuperado")
# plt.axis('off') # Oculta las marcas de los ejes para que se vea limpio
# plt.show()

import time

for paso, estado in enumerate(historial):
    imagen = estado.reshape((dimension, dimension))
    plt.imshow(imagen, cmap='gray_r', vmin=-1, vmax=1)
    plt.title(f"Iteración {paso}")
    plt.axis('off')
    plt.pause(0.2) # Pausa de 0.1 segundos entre cuadros
    plt.clf() # Limpia la figura para el siguiente paso