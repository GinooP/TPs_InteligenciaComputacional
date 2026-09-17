import numpy as np
import k_means as km
import SOM as s
import matplotlib.pyplot as plt

patrones_trn = np.loadtxt('Guia2/iris81_trn.csv', delimiter=',', skiprows=0)

patrones_tst = np.loadtxt('Guia2/iris81_tst.csv', delimiter=',', skiprows=0)

# K medias
patrones = patrones_trn[:,0:4]
nro_grupos = 5
epocas = 100
K, lotes_patrones = km.k_means(patrones, nro_grupos, epocas)
print(f"estos son los centroides:\n {K}")

# SOM
dim_matriz_neuronas = [6, 6]
eta = 0.9
vecindad = 3
cuadrado = False
epocas = [100, 200, 300]
matriz_neuronas = s.som(patrones, dim_matriz_neuronas, epocas, eta, vecindad, cuadrado, graf=False)



# Matriz de Contingencia

# # Comparacion con etiquetas reales, de referencia, clases conocidas, etc. ´
# # Comparacion entre diferentes soluciones de clustering. ´
# # Matriz de contingencia:
entradas = patrones_trn[:,0:4]
salidas = patrones_trn[:,4:7]

matriz_contingencia_SOM= s.tst_SOM(matriz_neuronas,entradas,salidas)

# # print(entradas)
# # print(salidas)
# #print(f"cantidad de entradas: {entradas.shape}")

# matriz_contingencia = km.tst_k_means(K, lotes_patrones, entradas, salidas)
# print(matriz_contingencia)

compactitudes,compactitud_global = km.calcular_compactitud(K,lotes_patrones,entradas)
print(f"compactitudes: \n {compactitudes}")
print(f"compactitud_global= {compactitud_global}")
# #graficamos los puntos en una dimension
# mascara_iris1= salidas[:,0] == 1
# mascara_iris2= salidas[:,1] == 1
# mascara_iris3= salidas[:,2] == 1
# #print(f"cantidad de patrones: {sum(mascara_iris1) + sum(mascara_iris2) + sum(mascara_iris3)}")
# plt.figure(0)
# plt.scatter(entradas[mascara_iris1,0],entradas[mascara_iris1,1],color='blue',label="iris1")
# plt.scatter(entradas[mascara_iris2,0],entradas[mascara_iris2,1],color='orange',label="iris2")
# plt.scatter(entradas[mascara_iris3,0],entradas[mascara_iris3,1],color='violet',label="iris3")
# plt.scatter(K[:,0],K[:,1],color="black",label="centroides")
# plt.title("Distribucion de las iris y los centroides")
# plt.legend()
# plt.show()


# --- AGREGAR AL FINAL DE TU SCRIPT ---

# 1. Inicializar matrices para frecuencias y clases
filas_som, cols_som = dim_matriz_neuronas
mapa_frecuencias = np.zeros((filas_som, cols_som))
mapa_votos_clases = np.zeros((filas_som, cols_som, 3)) # 3 clases de Iris

# 2. Calcular las frecuencias y votos iterando sobre los datos de entrenamiento
for i in range(len(entradas)):
    patron = entradas[i]
    clase_real = np.argmax(salidas[i,:]) # Devuelve 0, 1 o 2
    
    # Encontramos la neurona ganadora para este patrón
    pos_act = s.buscar_indice_activacion(matriz_neuronas, patron)
    
    # Incrementamos la frecuencia de activación
    mapa_frecuencias[pos_act[0], pos_act[1]] += 1
    
    # Registramos el "voto" de la clase para esa neurona
    mapa_votos_clases[pos_act[0], pos_act[1], clase_real] += 1

# 3. Graficar el Mapa 2D
plt.figure(figsize=(10, 8))

# Usamos imshow para crear una grilla de colores basada en las frecuencias
# cmap='Blues' mostrará celdas blancas para 0 activaciones y azul oscuro para muchas
img = plt.imshow(mapa_frecuencias, cmap='Blues', interpolation='nearest', origin='lower')
cbar = plt.colorbar(img)
cbar.set_label('Frecuencia de Activación')

# 4. Indicar la clase de cada neurona
# Definimos identificadores y colores para que resalten sobre el fondo
nombres_clases = ['1', '2', '3'] 
colores_texto = ['red', 'orange', 'green']

# Recorremos la grilla para escribir el número de la clase sobre las neuronas activas
for i in range(filas_som):
    for j in range(cols_som):
        if mapa_frecuencias[i, j] > 0: # Solo si la neurona se activó al menos una vez
            clase_ganadora = np.argmax(mapa_votos_clases[i, j])
            
            # Ponemos el texto en la coordenada (j, i) -> (X, Y)
            plt.text(j, i, nombres_clases[clase_ganadora], 
                     ha='center', va='center', 
                     color=colores_texto[clase_ganadora], 
                     fontweight='bold', fontsize=8)

plt.title('SOM en 2D: Frecuencia de Activación y Clases de Iris')
plt.xlabel('Columnas de Neuronas')
plt.ylabel('Filas de Neuronas')
plt.grid(False) # Apagamos la grilla por defecto de Matplotlib para que no moleste
plt.show(block=True)