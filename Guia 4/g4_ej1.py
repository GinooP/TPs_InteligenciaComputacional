import numpy as np
import SOM as S

patrones = np.loadtxt('Guia 4/circulo.csv', delimiter=',', skiprows=0)
dim_matriz_neuronas = (10,10)
epocas=1000
vecindad=3 #concideramos vecinas a las neuronas a 3 indices de distancia
eta=0.9
cuadrado=True
S.som(patrones, dim_matriz_neuronas, epocas, eta, vecindad,cuadrado)