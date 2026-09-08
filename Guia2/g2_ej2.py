import numpy as np
import perceptron_multicapa as mp
import time

inicio = time.perf_counter()

# Etapa de Entrenamiento
patrones = np.loadtxt('Guia2/concent_trn.csv', delimiter=',', skiprows=0)
patrones_tst = np.loadtxt('Guia2/concent_trn.csv', delimiter=',', skiprows=0)
eta = 0.05
epocas = 1000
capas = [6,1] # [6,6,1]
tol = 98

vector_capas, ratios, errores, errores_de_clasificacion = mp.generar_perceptron_multicapa(capas, patrones, eta, epocas, tol)

fin = time.perf_counter()
tiempo_transcurrido = fin - inicio
print(f"Tiempo de ejecución: {tiempo_transcurrido:.4f} segundos.")

ratio, error = mp.tst_perceptron_multicapa(patrones_tst, vector_capas, capas, plotear=True)

mp.graficar_resultados([eta], [errores], [errores_de_clasificacion], 6)