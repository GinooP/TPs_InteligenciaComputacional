import numpy as np
import matplotlib.pyplot as plt
import time

datos_or = np.loadtxt('Guia1/OR_trn.csv', delimiter=',')
datos_xor = np.loadtxt('Guia1/XOR_trn.csv', delimiter=',')

yd_or = datos_or[:,2]

yd_xor = datos_xor[:,2]

N = len(datos_or[:,0])
M = 3

x_or = -1 * np.ones((N,M))
x_or[:,1:M] = datos_or[:,0:M-1]

x_xor = -1 * np.ones((N,M))
x_xor[:,1:M] = datos_xor[:,0:M-1]

w_or = np.zeros((1,M)) # Vector de pesos sinápticos
w_xor = np.zeros((1,M)) # Vector de pesos sinápticos

rng = np.random.default_rng()

w_or = rng.random(M) - 0.5 # Setear los pesos entre [-0.5 0.5]
w_xor = rng.random(M) - 0.5 # Setear los pesos entre [-0.5 0.5]

tasa_de_aprendizaje = 0.1
maxit = 1
aciertos_or = 0
aciertos_xor = 0

plt.ion()

fig, ax = plt.subplots(figsize=(8, 5))
fig2, ax2 = plt.subplots(figsize=(8, 5))

for i in range(maxit):
    # Etapa de aprendizaje 
    for j in range(N):
        y = np.sign(np.dot(w_or,x_or[j,:]))
        w_or = w_or + tasa_de_aprendizaje/2 * (yd_or[j] - y)*x_or[j,:]

        ax.clear() # Limpiamos la recta anterior
        ax.set_title(f"Evolución de frontera OR (Iteración {i}, Dato {j})")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        
        # Opcional pero recomendado: fijar los ejes para que el gráfico no "salte"
        ax.set_xlim(-1.5, 1.5) 
        ax.set_ylim(-1.5, 1.5)
        
        # Dibujar los puntos de datos (para ver cómo la recta los intenta separar)
        # Usamos c=yd_or para que los pinte de distinto color según la clase esperada
        ax.scatter(x_or[:,1], x_or[:,2], c=yd_or, cmap='bwr', s=100, edgecolors='k')
        
        # Calcular y dibujar la recta usando la COLUMNA 1
        x2_frontera_or = (w_or[0]/w_or[2]) - (w_or[1]/w_or[2]) * x_or[:,1]
        ax.plot(x_or[:,1], x2_frontera_or, color='black', linewidth=2)
        

        y = np.sign(np.dot(w_xor,x_xor[j,:]))
        w_xor = w_xor + tasa_de_aprendizaje/2 * (yd_xor[j] - y)*x_xor[j,:]

        ax2.clear() # Limpiamos la recta anterior
        ax2.set_title(f"Evolución de frontera XOR (Iteración {i}, Dato {j})")
        ax2.set_xlabel("X1")
        ax2.set_ylabel("X2")
        
        # Opcional pero recomendado: fijar los ejes para que el gráfico no "salte"
        ax2.set_xlim(-1.5, 1.5) 
        ax2.set_ylim(-1.5, 1.5)
        
        # Dibujar los puntos de datos (para ver cómo la recta los intenta separar)
        # Usamos c=yd_xor para que los pinte de distinto color según la clase esperada
        ax2.scatter(x_xor[:,1], x_xor[:,2], c=yd_xor, cmap='bwr', s=100, edgecolors='k')

        x2_frontera_xor = (w_xor[0]/w_xor[2]) - (w_xor[1]/w_xor[2])*x_xor[:,1]
        ax2.plot(x_xor[:,1], x2_frontera_xor, color='black', linewidth=2)
        
        plt.pause(0.5)

    # Etapa de winrate
    aciertos_or = 0
    for j in range(N):
        y = np.sign(np.dot(w_or,x_or[j,:]))
        if (yd_or[j] - y) == 0:
            aciertos_or += 1
    ratio_or = aciertos_or / N

    aciertos_xor = 0
    for j in range(N):
        y = np.sign(np.dot(w_xor,x_xor[j,:]))
        if (yd_xor[j] - y) == 0:
            aciertos_xor += 1
    ratio_xor = aciertos_xor / N


print(f"Ratio OR: {ratio_or}")
print(f"Ratio XOR: {ratio_xor}")

plt.ioff()
plt.show()