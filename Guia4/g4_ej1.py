import numpy as np
import SOM as S

patrones = np.loadtxt('Guia4/te.csv', delimiter=',', skiprows=0)
dim_matriz_neuronas = (3,4)
epocas=[100,600,300] #epocas=[50,50,50], #vecindad=3, eta=0.9 anda bien muy bien
vecindad=3
eta=0.9
cuadrado=False

S.som(patrones, dim_matriz_neuronas, epocas, eta, vecindad,cuadrado, graf=True)