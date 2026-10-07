# Ejercicio 1: Implemente un algoritmo de optimizacion por enjambre de particulas
# y utilicelo para encontrar el minimo global de las funciones del Ejercicio 1 de
# la Guia de trabajos practicos 6.
# Compare los resultados en relacion a los obtenidos con algoritmos geneticos,
# en terminos de las soluciones encontradas y la velocidad de convergencia.

import numpy as np
import algoritmos as alg

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

# Ejecutar PSO
sol = alg.algoritmo_enjambre_particulas(
       cant_particulas,     
       limites_dimensiones, 
       aceleracion,         
       funcion,             
       cant_vecinos,        
       aptitud_requerida,   
       maximo_iteraciones   
)

print(f"\nResultado final (Mejor Posición): f({sol}) = {f1(sol)}")

print("\n --- Segunda Funcion ---\n")

f2 = lambda x,y: (x**2 + y**2)**(0.25)*(np.sin(50 * (x**2 + y**2)**(0.1))**2 + 1)

def f_aptitud2(val):
    # Evaluamos la función. Como PSO minimiza, devolvemos el valor real 
    # para que busque el fondo del valle (que sabemos que es 0)
    valores = f2(val[:, 0], val[:, 1])
    return valores

cant_particulas = 100
limites_dimensiones = np.array([[-100,-100], [100,100]]) 
aceleracion = [1.5, 1.5] 
funcion = f_aptitud2
cant_vecinos = cant_particulas
aptitud_requerida = 0.001
maximo_iteraciones = 1000

# Ejecutar PSO
sol = alg.algoritmo_enjambre_particulas(
       cant_particulas,     
       limites_dimensiones, 
       aceleracion,         
       funcion,             
       cant_vecinos,        
       aptitud_requerida,   
       maximo_iteraciones   
)

print(f"\nResultado final (Mejor Posición): f({sol}) = {f2(sol[0],sol[1])}")