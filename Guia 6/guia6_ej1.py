import numpy as np
import math
import computacion_evolutiva as evo


def func_aptitud(poblacion):

    def decodificar_binario_con_signo (arreglo):
        valor=0
        N=len(arreglo)
        for i in range(N-1):

            valor += arreglo[i]*(2**i)

        if(arreglo[-1]==1):
            valor = -valor

        return valor
    
    N = len(poblacion[:,0])
    f = lambda x: (-1)*(-x * math.sin(abs(x)**0.5))
    fitnes = []
    for i in range(N):

        valor=decodificar_binario_con_signo(poblacion[i,:])
        #print(f"valor= {valor}")
        fitnes.append(f(valor))

    return fitnes

cant_individuos=30
cant_bits=10
cant_progenitores=5
prob_mutacion=0.1#probabilidad de mutacion
elitismo=True

aptitud_requerida=415 #obtenido por inspeccion visual en geogebra (podria estimarlo con biseccion en octave)

solucion,mejores_fitnes,peores_fitnes = evo.algoritmo_evolutivo(cant_individuos,cant_bits,func_aptitud,
                                                                cant_progenitores,aptitud_requerida,prob_mutacion,elitismo)

print(solucion)
print(mejores_fitnes[-1])



