import numpy as np
import SOM as S

patrones = np.loadtxt('Guia 4/te.csv', delimiter=',', skiprows=0)
dim_matriz_neuronas = (10,10)
epocas=[250,500,250] #epocas=[50,50,50], #vecindad=3, eta=0.9 anda bien muy bien
vecindad=3
eta=0.9
cuadrado=False

S.som(patrones, dim_matriz_neuronas, epocas, eta, vecindad,cuadrado)