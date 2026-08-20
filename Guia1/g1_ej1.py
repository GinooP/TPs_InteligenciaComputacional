import numpy as np
import trn_perceptron as trn
import matplotlib.pyplot as plt
import time

# Etapa de Entrenamiento
datos = np.loadtxt('Guia1/OR_trn.csv', delimiter=',', skiprows=0)
tasa_de_aprendizaje = 0.1
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

# Grafico de observación
x1 = x_tst[:,1]
x2 = x_tst[:,2]

# Separar los datos en clases
clase_positiva = y_d_tst == 1
clase_negativa = y_d_tst == -1

plt.figure(figsize=(8,6))

plt.scatter(x1[clase_positiva], x2[clase_positiva], c='blue', marker='o', label='Clase +1', s=100)
plt.scatter(x1[clase_negativa], x2[clase_negativa], c='red', marker='x', label='Clase -1', s=100)

x_recta = np.linspace(min(x1) - 0.5, max(x1) + 0.5, 100)
y_recta = (w[0] - w[1] * x_recta) / w[2]

plt.plot(x_recta, y_recta, 'g-', label='Recta de decisión')

plt.title('Clasificación Perceptrón - Compuerta OR')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(min(x1) - 0.5, max(x1) + 0.5)
plt.ylim(min(x2) - 0.5, max(x2) + 0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()