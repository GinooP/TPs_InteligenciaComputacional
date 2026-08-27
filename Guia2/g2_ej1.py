import numpy as np
import matplotlib.pyplot as plt
import perceptron_multicapa as mp

# Etapa de Entrenamiento
patrones = np.loadtxt('Guia1/XOR_trn.csv', delimiter=',', skiprows=0)
eta = 0.2
epocas = 500
capas = [2,1]
tol = 100

v,r,e = mp.generar_perceptron_multicapa(capas, patrones, eta, epocas, tol)

print(e)