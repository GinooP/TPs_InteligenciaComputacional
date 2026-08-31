import numpy as np
import matplotlib.pyplot as plt
import time
import perceptron_multicapa as mp

# Etapa de Entrenamiento
patrones = np.loadtxt('Guia2/concent_trn.csv', delimiter=',', skiprows=0)
eta = 0.1
epocas = 100
capas = [40,40,1]
tol = 1

mp.generar_perceptron_multicapa(capas, patrones, eta, epocas, tol)