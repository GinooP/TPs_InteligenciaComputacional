import numpy as np
import algoritmos as alg
from scipy.optimize import minimize
import time as tm

# ------------------------- INCISO 1 -------------------------

def decodificar(poblacion, genes):
    x_min = -512
    x_max = 512
    b = poblacion.shape[1]
    cant_individuos = len(poblacion[:,0])
    x_entero = np.zeros(cant_individuos)

    for j in range(cant_individuos):
        for i in range(b):
            x_entero[j] += poblacion[j,i]*(2**i)

    fenotipos = x_min + x_entero*((x_max - x_min)/(2**b - 1))
    return fenotipos

f1 = lambda x: -x*np.sin(np.sqrt(np.abs(x)))

def f_aptitud(fenotipos):
    valores = f1(fenotipos)
    aptitudes = -valores
    return aptitudes


cant_individuos = 50
genes = [10]
aptitud_requerida = 418
itmax = 200
cant_progenitores = 10
tasa_de_mutacion = [0.3, 'individuo']
tasa_de_brecha = 0.3
graf = True

individuo = alg.algoritmo_genetico(
    cant_individuos, genes, aptitud_requerida, itmax, 
    tasa_de_mutacion=tasa_de_mutacion,
    f_decodificacion=decodificar,
    f_aptitud=f_aptitud,
    graf=graf
    )

fenotipos = decodificar(np.array([individuo]),genes)
value = f1(fenotipos)
print(f'Individuo = {individuo} -> f(x={fenotipos[0]:.4f}) = {value[0]:.4f}\n')


# ------------------------- INCISO 2 -------------------------

def decodificar2(poblacion, genes):
    min_val = -100
    max_val = 100
    cant_genes = len(genes)
    cant_individuos = poblacion.shape[0]

    fenotipos = np.zeros((cant_individuos,cant_genes))
    b_prev = 0
    for j in range(cant_genes):

        x_entero = 0
        b = genes[j]
        for i in range(0,b):
            x_entero += poblacion[:,i+b_prev]*(2**i)

        fenotipos[:,j] = min_val + x_entero*((max_val - min_val)/(2**genes[j] - 1))
        b_prev += genes[j]

    return fenotipos

f2 = lambda x,y: (x**2 + y**2)**(0.25)*(np.sin(50 * (x**2 + y**2)**(0.1))**2 + 1)

def f_aptitud2(fenotipos):

    valores = f2(fenotipos[:,0],fenotipos[:,1])
    val_max = np.max(valores)
    bias = 0.001
    aptitudes = (val_max - valores)/(val_max + bias)

    return aptitudes


cant_individuos = 60
genes = [10, 10]
aptitud_requerida = 0.999
itmax = 2000
cant_progenitores = 10
tasa_de_mutacion = [0.3, 'individuo']
tasa_de_brecha = 0.3
graf = True

individuo = alg.algoritmo_genetico(
    cant_individuos, genes, aptitud_requerida, itmax, 
    tasa_de_mutacion=tasa_de_mutacion,
    f_decodificacion=decodificar2,
    f_aptitud=f_aptitud2,
    graf=graf
    )

fenotipos = decodificar2(np.array([individuo]), genes)
value = f2(fenotipos[0,0], fenotipos[0,1])
print(f'Individuo = {individuo} -> f(x={fenotipos[0,0]:.4f},y={fenotipos[0,1]:.4f}) = {value:.4f} \n')

# # ------------------------- Metodo de Gradiente Descendiente -------------------------

# # SciPy asume que la entrada es un único arreglo (v). 
# # Para f1, v[0] es x.
# f1_scipy = lambda v: -v[0] * np.sin(np.sqrt(np.abs(v[0])))

# # Para f2, v[0] es x y v[1] es y.
# f2_scipy = lambda v: (v[0]**2 + v[1]**2)**(0.25) * (np.sin(50 * (v[0]**2 + v[1]**2)**(0.1))**2 + 1)

# # Definimos los límites según la consigna
# limites_f1 = [(-512, 512)]
# # Para f2 necesitamos dos límites, uno para x y otro para y
# limites_f2 = [(-100, 100), (-100, 100)] 

# # Variables para guardar estadísticas de múltiples pruebas
# resultados_gd_f1 = []
# tiempos_gd_f1 = []
# resultados_gd_f2 = []
# tiempos_gd_f2 = []

# # Ejecutamos el gradiente descendiente 30 veces para ver su dependencia del inicio
# for i in range(30):
#     # 1. Elegir punto de partida aleatorio
#     x_inicial_f1 = np.random.uniform(-512, 512, 1)
#     # Para f2 necesitamos generar 2 valores aleatorios
#     x_inicial_f2 = np.random.uniform(-100, 100, 2) 

#     # method='L-BFGS-B':
#     #   una variante de gradiente que permite establecer límites espaciales 
#     #   que aproxima el gradiente numéricamente de forma automática

#     # 2. Ejecutar la minimización para f1
#     inicio = tm.perf_counter()
#     res_f1 = minimize(f1_scipy, x_inicial_f1, method='L-BFGS-B', bounds=limites_f1)
#     fin = tm.perf_counter()
    
#     resultados_gd_f1.append(res_f1.fun)
#     tiempos_gd_f1.append(fin - inicio)

#     # 3. Ejecutar la minimización para f2
#     inicio = tm.perf_counter()
#     res_f2 = minimize(f2_scipy, x_inicial_f2, method='L-BFGS-B', bounds=limites_f2)
#     fin = tm.perf_counter()

#     resultados_gd_f2.append(res_f2.fun)
#     tiempos_gd_f2.append(fin - inicio)

# print(' Metodo de Gradiente Descendiente con libreria SciPy \n')
# print(' FUNCION INCISO 1:')
# print("Mínimos encontrados (GD) en cada corrida:")
# for i, val in enumerate(resultados_gd_f1):
#     print(f"Corrida {i+1}: {val:.4f}")
# print(f"Desvío estándar de los resultados: {np.std(resultados_gd_f1):.4f}")
# print(f"Tiempo promedio por ejecución: {np.mean(tiempos_gd_f1):.6f} seg\n")

# print(' FUNCION INCISO 2:')
# print("Mínimos encontrados (GD) en cada corrida:")
# for i, val in enumerate(resultados_gd_f2):
#     print(f"Corrida {i+1}: {val:.4f}")
# print(f"Desvío estándar de los resultados: {np.std(resultados_gd_f2):.4f}")
# print(f"Tiempo promedio por ejecución: {np.mean(tiempos_gd_f2):.6f} seg")