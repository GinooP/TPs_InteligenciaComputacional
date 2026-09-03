import numpy as np
import matplotlib.pyplot as plt
import time
import perceptron_multicapa as mp
import perceptron_multicapa_tst as pm_tst

class Capa:
    def __init__(self, W, y, delta):
        self.W = W # Matriz de pesos
        self.y = y # Vector de salidas
        self.delta = delta # Vector de errores

# Etapa de Entrenamiento
patrones = np.loadtxt('Guia2/concent_trn.csv', delimiter=',', skiprows=0)
eta = 0.01
epocas = 100
capas = [6,1]
tol = 1
winout=False

vector_capas,error_clasificacion=mp.generar_perceptron_multicapa(capas, patrones, eta, epocas, tol,winout)

patrones = np.loadtxt('Guia2/concent_tst.csv', delimiter=',', skiprows=0)
vector_capas = pm_tst.test_perceptron_multicapa(capas,vector_capas,patrones,winout)

### graficamos ##########
M = len(patrones[0,:]) - capas[-1] + 1# cantidad de entradas + bias
N = len(patrones[:,0]) # cantidad de patrones

# Matriz de Entradas
x = -1 * np.ones((N,M))
x[:,1:M] = patrones[:,0:M-1]

# Vector de salidas esperadas
d = patrones[:,M-1:len(patrones[0,:])]#concideramos que puede haber mas salidas

# plt.figure(1)
# plt.plot(patrones[0,:],patrones[1,:],c=d)
# plt.show()