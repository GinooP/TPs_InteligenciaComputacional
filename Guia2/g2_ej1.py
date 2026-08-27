import numpy as np
import matplotlib.pyplot as plt
import perceptron_multicapa as mp

# Etapa de Entrenamiento
patrones = np.loadtxt('Guia1/XOR_trn.csv', delimiter=',', skiprows=0)
eta = 0.01
epocas = 500
capas = [2,1]
tol = 100

vector_capas, ratios, errores = mp.generar_perceptron_multicapa(capas, patrones, eta, epocas, tol)


patrones_tst = np.loadtxt('Guia1/XOR_tst.csv', delimiter=',', skiprows=0)
ratio, error = mp.tst_perceptron_multicapa(patrones_tst, vector_capas, capas)