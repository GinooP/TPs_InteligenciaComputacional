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
patrones = np.loadtxt('Guia2/iris81_trn.csv', delimiter=',', skiprows=0)
eta = 0.05
epocas = 1000
capas = [10,10,3]
tol = 0.99
winout=True
vector_capas,error_clasificacion =mp.generar_perceptron_multicapa(capas, patrones, eta, epocas, tol, winout)
patrones = np.loadtxt('Guia2/iris81_tst.csv', delimiter=',', skiprows=0)
print("############################### Test ################################")
#mp.tst_perceptron_multicapa(patrones,vector_capas,capas,winout)
pm_tst.test_perceptron_multicapa(capas,vector_capas,patrones,winout)


