# Ejercicio 1: Implemente un algoritmo de optimizacion por enjambre de particulas
# y utilicelo para encontrar el minimo global de las funciones del Ejercicio 1 de
# la Guia de trabajos practicos 6.
# Compare los resultados en relacion a los obtenidos con algoritmos geneticos,
# en terminos de las soluciones encontradas y la velocidad de convergencia.

import numpy as np
import time as tm
import algoritmos as alg
import sys
import os

ruta_padre = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_padre)

from Guia6 import algoritmos as alg_gen

print("\n ####### Enjambre de Particulas #######")
print("\n --- Primera Funcion ---\n")

f1 = lambda x: -x * np.sin(np.sqrt(np.abs(x)))

def f_aptitud(val):
    valores = f1(val)

    aptitudes = np.sum(valores, axis=1)

    return aptitudes

cant_particulas = 30
limites_dimensiones = np.array([[-512], [512]]) 
aceleracion = [1.5, 1.5]
funcion = f_aptitud
cant_vecinos = 3
aptitud_requerida = -418
maximo_iteraciones = 150

sol1,ttotal1 = alg.algoritmo_enjambre_particulas(
       cant_particulas,     
       limites_dimensiones, 
       aceleracion,         
       funcion,             
       cant_vecinos,        
       aptitud_requerida,   
       maximo_iteraciones   
)

print(f'Resultado final: f(x={sol1[0]:.4f}) = {f1(sol1[0]):.4f} \n')

print("\n --- Segunda Funcion ---\n")

f2 = lambda x,y: (x**2 + y**2)**(0.25)*(np.sin(50 * (x**2 + y**2)**(0.1))**2 + 1)

def f_aptitud2(val):
    valores = f2(val[:, 0], val[:, 1])
    return valores

cant_particulas = 100
limites_dimensiones = np.array([[-100,-100], [100,100]]) 
aceleracion = [0.5, 0.5] 
funcion = f_aptitud2
cant_vecinos = cant_particulas
aptitud_requerida = 0.001
maximo_iteraciones = 1000

# Ejecutar PSO
sol2,ttotal2 = alg.algoritmo_enjambre_particulas(
       cant_particulas,     
       limites_dimensiones, 
       aceleracion,         
       funcion,             
       cant_vecinos,        
       aptitud_requerida,   
       maximo_iteraciones   
)

print(f'Resultado final: f(x={sol2[0]:.4f},y={sol2[1]:.4f}) = {f2(sol2[0],sol2[1]):.4f} \n')

print("\n ####### Algoritmo Genetico #######")

print("\n --- Primera Funcion ---\n")

def decodificar(individuo):
    x_min = -512
    x_max = 512
    x_entero = 0
    b = len(individuo)

    for i in range(0,b):
        x_entero += individuo[i]*(2**i)

    x = x_min + x_entero*((x_max - x_min)/(2**b - 1))
    return x

def f_aptitud(fenotipos):
    valores = f1(fenotipos[:,0])
    aptitudes = -valores
    return aptitudes

individuos = [10, 50]
itmax = 200
aptitud_requerida = 418
cant_variables = 1
graf = False

inicio = tm.perf_counter()
individuo1 = alg_gen.algoritmo_genetico(
    individuos, 
    decodificar, 
    f_aptitud, 
    alg_gen.mecanismo_ventanas, 
    alg_gen.mutacion, 
    alg_gen.cruzas_simples, 
    alg_gen.elitismo, 
    aptitud_requerida, 
    itmax, 
    cant_variables,
    graf
)
fin = tm.perf_counter()

ttotal3 = fin - inicio

fenotipo1 = decodificar(individuo1)
value1 = f1(fenotipo1)
print(f'\nTiempo de busqueda = {ttotal3:.6f} seg')
print(f'Resultado final: f({fenotipo1:.4f}) = {value1:.4f}\n')

print("\n --- Segunda Funcion ---\n")


def decodificar2(individuo):
    min_val = -100
    max_val = 100
    x_entero = 0
    y_entero = 0
    b = len(individuo) // 2

    for i in range(0,b):
        x_entero += individuo[i]*(2**i)
        
    for i in range(0,b):
        y_entero += individuo[i+b]*(2**i)

    x = min_val + x_entero*((max_val - min_val)/(2**b - 1))
    y = min_val + y_entero*((max_val - min_val)/(2**b - 1))

    return np.array([x,y])

def f_aptitud2(fenotipos):

    valores = f2(fenotipos[:,0],fenotipos[:,1])
    val_max = np.max(valores)
    bias = 0.001
    aptitudes = (val_max - valores)/(val_max + bias)

    return aptitudes


individuos = [60, 200]
itmax = 200
aptitud_requerida = 0.99
cant_variables = 2
graf = False

inicio = tm.perf_counter()
individuo2 = alg_gen.algoritmo_genetico(
    individuos, 
    decodificar2, 
    f_aptitud2, 
    alg_gen.mecanismo_ventanas, 
    alg_gen.mutacion, 
    alg_gen.cruzas_simples, 
    alg_gen.elitismo, 
    aptitud_requerida, 
    itmax, 
    cant_variables, 
    graf
)
fin = tm.perf_counter()

ttotal4 = fin - inicio

fenotipo2 = decodificar2(individuo2)
value2 = f2(fenotipo2[0], fenotipo2[1])
print(f'\nTiempo de busqueda = {ttotal4:.6f} seg')
print(f'Resultado final: f(x={fenotipo2[0]:.4f},y={fenotipo2[1]:.4f}) = {value2:.4f} \n')


# ==============================================================================
#                              CUADRO COMPARATIVO
# ==============================================================================

# Valores finales para que el código quede limpio en la tabla
val_pso1 = f1(sol1[0])
val_pso2 = f2(sol2[0], sol2[1])

print("\n" + "="*100)
print(f"{'CUADRO COMPARATIVO DE RENDIMIENTO':^90}")
print("="*100)
print(f"{'Función':<16} | {'Algoritmo':<22} | {'Mejor Valor f(x)':<15} | {'Tiempo (seg)':<12} | {'Solución'}")
print("-" * 100)

# Fila 1: Función 1
print(f"{'F1 (1 variable)':<16} | {'Enjambre Partículas':<22} | {val_pso1:<16.4f} | {ttotal1:<12.6f} | x = {sol1[0]:.4f}")
print(f"{'':<16} | {'Algoritmo Genético':<22} | {value1:<16.4f} | {ttotal3:<12.6f} | x = {fenotipo1:.4f}")
print("-" * 100)

# Fila 2: Función 2
print(f"{'F2 (2 variables)':<16} | {'Enjambre Partículas':<22} | {val_pso2:<16.4f} | {ttotal2:<12.6f} | x = {sol2[0]:.4f}, y = {sol2[1]:.4f}")
print(f"{'':<16} | {'Algoritmo Genético':<22} | {value2:<16.4f} | {ttotal4:<12.6f} | x = {fenotipo2[0]:.4f}, y = {fenotipo2[1]:.4f}")
print("="*100 + "\n")


# ====================================================================================================
#                             CUADRO COMPARATIVO DE RENDIMIENTO                             
# ====================================================================================================
# Función          | Algoritmo              | Mejor Valor f(x) | Tiempo (seg) | Solución
# ----------------------------------------------------------------------------------------------------
# F1 (1 variable)  | Enjambre Partículas    | -418.8377        | 0.039583     | x = 422.0413
#                  | Algoritmo Genético     | -418.8706        | 0.013112     | x = 421.9120
# ----------------------------------------------------------------------------------------------------
# F2 (2 variables) | Enjambre Partículas    | 0.0009           | 0.554650     | x = 0.0000, y = -0.0000
#                  | Algoritmo Genético     | 0.0966           | 0.180364     | x = -0.0024, y = 0.0063
# ====================================================================================================
# F1 (1 variable)  | Enjambre Partículas    | -418.7090        | 0.033207     | x = 419.4950
#                  | Algoritmo Genético     | -418.8706        | 0.006556     | x = 421.9120
# ----------------------------------------------------------------------------------------------------
# F2 (2 variables) | Enjambre Partículas    | 0.0010           | 0.592546     | x = -0.0000, y = 0.0000
#                  | Algoritmo Genético     | 0.0959           | 0.280485     | x = -0.0049, y = 0.0065
# ====================================================================================================
# F1 (1 variable)  | Enjambre Partículas    | -418.9512        | 0.025387     | x = 420.4677
#                  | Algoritmo Genético     | -418.8706        | 0.004944     | x = 421.9120
# ----------------------------------------------------------------------------------------------------
# F2 (2 variables) | Enjambre Partículas    | 0.0008           | 0.574049     | x = -0.0000, y = -0.0000
#                  | Algoritmo Genético     | 0.0878           | 0.153641     | x = -0.0066, y = -0.0040
# ====================================================================================================
# F1 (1 variable)  | Enjambre Partículas    | -418.0860        | 0.037194     | x = 418.3009
#                  | Algoritmo Genético     | -418.8415        | 0.005302     | x = 419.9101
# ----------------------------------------------------------------------------------------------------
# F2 (2 variables) | Enjambre Partículas    | 0.0007           | 0.526900     | x = -0.0000, y = 0.0000
#                  | Algoritmo Genético     | 0.0944           | 0.323698     | x = -0.0078, y = -0.0021
# ====================================================================================================
# F1 (1 variable)  | Enjambre Partículas    | -418.8079        | 0.030784     | x = 419.7908
#                  | Algoritmo Genético     | -418.4481        | 0.007507     | x = 418.9091
# ----------------------------------------------------------------------------------------------------
# F2 (2 variables) | Enjambre Partículas    | 0.0007           | 0.595723     | x = -0.0000, y = 0.0000
#                  | Algoritmo Genético     | 0.0563           | 0.460628     | x = -0.0016, y = 0.0023
# ====================================================================================================
