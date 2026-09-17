import numpy as np
import matplotlib.pyplot as plt
import k_means as km

# 1. Carga de datos
patrones_trn = np.loadtxt('Guia2/iris81_trn.csv', delimiter=',', skiprows=0)
entradas = patrones_trn[:, 0:4] 
salidas = patrones_trn[:, 4:7] 

K_values = range(2,31)
epocas = 100

compactitudes = [] 

# 2. Iteración sobre los valores de K
for k in K_values:

    K_centroides, lotes_patrones = km.k_means(entradas, k, epocas)

    compactitudes_por_lote, compactitud_global = km.calcular_compactitud(K_centroides, lotes_patrones, entradas)

    matriz_contingencia = km.tst_k_means(K_centroides, lotes_patrones, entradas, salidas)
    compactitudes.append(compactitud_global)

# --- 3. CÓDIGO PARA GRAFICAR (MÉTODO DEL CODO) ---
plt.figure(figsize=(8, 5))
plt.plot(K_values, compactitudes, marker='o', linestyle='-', color='b', markersize=8)

plt.title('Método del Codo para K-Medias (Dataset Iris)', fontsize=14)
plt.xlabel('Número de clusters (k)', fontsize=12)
plt.ylabel('Compactitud Global (Inercia)', fontsize=12)

# Asegurar que el eje X muestre todos los valores enteros de K
plt.xticks(K_values)
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()