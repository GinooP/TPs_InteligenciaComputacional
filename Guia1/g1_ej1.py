import numpy as np
import trn_perceptron as trn

# Etapa de Entrenamiento
datos = np.loadtxt('Guia1/OR_trn.csv', delimiter=',', skiprows=0)
tasa_de_aprendizaje = 0.5
maxit = 1
w, ratio, matriz_w = trn.trn_perceptron(datos, tasa_de_aprendizaje, maxit)

print(f"El ratio de aciertos es de: {ratio*100} %")

# Etapa de Validacion

datos_tst = np.loadtxt('Guia1/OR_tst.csv', delimiter=',', skiprows=0)

N = len(datos_tst[:,0]) # Cantidad de patrones (pruebas)
M = len(datos_tst[0,:]) # Cantidad de entradas +

y_d_tst = datos_tst[:,M-1] # Vector con salidas esperadas

x_tst = -1 * np.ones((N,M))
x_tst[:,1:M] = datos_tst[:,0:M-1]

aciertos_tst = 0
for j in range(N):
    y = np.sign(np.dot(w,x_tst[j,:]))
    if (y_d_tst[j] - y) == 0:
        aciertos_tst += 1
ratio_tst = aciertos_tst / N

print(f"El ratio de aciertos del test es de: {ratio_tst*100} %")
