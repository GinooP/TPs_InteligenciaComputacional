import numpy as np
import matplotlib.pyplot as plt
import time
import perceptron_multicapa as mp

# Etapa de Entrenamiento
patrones = np.loadtxt('Guia1/concent-trn.csv', delimiter=',', skiprows=0)
eta = 0.1
epocas = 100
capas = [2,1]
tol = 1

mp.generar_perceptron_multicapa(capas, patrones, eta, epocas, tol)