import time
import numpy as np

class grafo:
    def __init__(self, V, E):
        self.V = V
        self.E = E

def coloniaHormigas(G, num_hormigas, tmax, alpha, beta, rho, Q, metodo_actualizacion):
    #G: grafo con nodos del camino
    #num_hormigas: número de hormigas
    #tmax: número de iteraciones
    #alpha: importancia de la feromona
    #beta: importancia del costo
    #rho: tasa de evaporación
    #Q: cantidad de feromona depositada

    N=len(G.V) #Número de nodos en el grafo
    #Inicialización de feromonas.
    feromonas = np.ones((len(G.V), len(G.V))) #Matriz de feromonas inicializada a 1
    #Representa la cantidad de feromona en cada arista del grafo.

    mejor_camino = None
    mejor_costo = 9999999

    for t in range(tmax):
        caminos = [] #Vector de vectores.
        costos = []

        for hormiga in range(num_hormigas):
            camino_actual = [np.random.choice(G.V)] #Comenzar desde un nodo aleatorio
            while len(camino_actual) < len(G.V):
                nodo_actual = camino_actual[-1]
                vecinos = []
                for v in G.V:
                    if v not in camino_actual:
                        vecinos.append(v)

                probabilidades = []

                for vecino in vecinos:
                    costo_arista = G.E[nodo_actual][vecino] #Costo de la arista
                    feromona = feromonas[nodo_actual][vecino]
                    probabilidad = (feromona ** alpha) * ((1 / costo_arista) ** beta)
                    probabilidades.append(probabilidad)

                probabilidades = np.array(probabilidades)
                probabilidades /= probabilidades.sum() #Normalizar

                siguiente_nodo = np.random.choice(vecinos, p=probabilidades)
                camino_actual.append(siguiente_nodo)

            costo_total = 0
            for i in range(len(camino_actual) - 1):
                costo_total+= G.E[camino_actual[i]][camino_actual[i + 1]]
            costo_total += G.E[camino_actual[-1]][camino_actual[0]] #Volver al nodo inicial para el problema del viajero
            caminos.append(camino_actual)
            costos.append(costo_total)

            if costo_total < mejor_costo:
                mejor_costo = costo_total
                mejor_camino = camino_actual

        #print(f"Iteración {t+1}], Mejor Costo: {mejor_costo}, Peor Costo: {max(costos)}")
        #Criterio de parada: si el peor camino es igual al mejor, entonces todas las hormigas ya encontraron el mismo
        if np.abs(max(costos)-min(costos))/min(costos)<1e-6: 
            print(f"## Convergencia alcanzada en la iteración {t+1}.")
            break

        #Actualización de feromonas
        feromonas *= (1 - rho)  #Evaporación
        for i in range(num_hormigas):
            for j in range(len(caminos[i]) - 1):
                nodo_a = caminos[i][j]
                nodo_b = caminos[i][(j + 1)%len(caminos[i])] #El % asegura de que el índice sea circular
                #Depositar feromona:
                if metodo_actualizacion == "uniforme":
                    feromonas[nodo_a][nodo_b] += Q #Actualización uniforme
                elif metodo_actualizacion == "global":
                    feromonas[nodo_a][nodo_b] += Q / costos[i] #Actualización global
                elif metodo_actualizacion == "local":
                    feromonas[nodo_a][nodo_b] += Q / G.E[nodo_a][nodo_b] #Actualización local

    return mejor_camino, mejor_costo

# E=np.loadtxt("Guia7/gr17.csv",delimiter=",")
# G=grafo(V=list(range(len(E))),E=E)

#Aumentando alfa, se alcanza más seguido el criterio de parada porque más hormigas siguen las feromonas.
#A su vez, esto hace más probable que se quede atrapado en un camino óptimo local.
# for i in range(1, 5):
#     print(f"\n######ITERACIÓN {i}#######")
#     ini=time.perf_counter()
#     CH_global=coloniaHormigas(G,num_hormigas=20,tmax=400,alpha=1+i*0.3,beta=2.5,rho=0.1,Q=10,metodo_actualizacion="global")
#     fin=time.perf_counter()
#     tiempo_total = fin - ini
#     print(f"###GLOBAL: El algoritmo demoró: {tiempo_total:.4f} segundos")
#     print("Mejor camino:", CH_global[0])
#     print("Mejor costo:", CH_global[1])

#     ini=time.perf_counter()
#     CH_local=coloniaHormigas(G,num_hormigas=20,tmax=400,alpha=1+i*0.3,beta=2.5,rho=0.1,Q=10,metodo_actualizacion="local")
#     fin=time.perf_counter()
#     tiempo_total = fin - ini
#     print(f"###LOCAL: El algoritmo demoró: {tiempo_total:.4f} segundos")
#     print("Mejor camino:", CH_local[0])
#     print("Mejor costo:", CH_local[1])

#     ini=time.perf_counter()
#     CH_uniforme=coloniaHormigas(G,num_hormigas=20,tmax=400,alpha=1+i*0.3,beta=2.5,rho=0.1,Q=10,metodo_actualizacion="uniforme")
#     fin=time.perf_counter()
#     tiempo_total = fin - ini
#     print(f"###UNIFORME: El algoritmo demoró: {tiempo_total:.4f} segundos")
#     print("Mejor camino:", CH_uniforme[0])
#     print("Mejor costo:", CH_uniforme[1])


E=np.loadtxt("Guia7/gr17.csv",delimiter=",")
G=grafo(V=list(range(len(E))),E=E)

#Aumentando alfa, se alcanza más seguido el criterio de parada porque más hormigas siguen las feromonas.
#A su vez, esto hace más probable que se quede atrapado en un camino óptimo local.
# Imprimimos el encabezado de la tabla
print(f"\n{'Iter':<4} | {'Alpha':<5} | {'Método':<10} | {'Tiempo (s)':<10} | {'Costo':<8} | {'Mejor Camino'}")
print("-" * 110)

for i in range(1, 5):
    alpha_val = 1 + i * 0.3
    
    # GLOBAL
    ini=time.perf_counter()
    CH_global=coloniaHormigas(G,num_hormigas=20,tmax=400,alpha=alpha_val,beta=2.5,rho=0.1,Q=10,metodo_actualizacion="global")
    fin=time.perf_counter()
    # Convertimos los nodos (np.int64) a int de Python solo para la impresión
    camino_g = [int(nodo) for nodo in CH_global[0]]
    print(f"{i:<4} | {alpha_val:<5.1f} | {'Global':<10} | {fin-ini:<10.4f} | {CH_global[1]:<8.2f} | {camino_g}")

    # LOCAL
    ini=time.perf_counter()
    CH_local=coloniaHormigas(G,num_hormigas=20,tmax=400,alpha=alpha_val,beta=2.5,rho=0.1,Q=10,metodo_actualizacion="local")
    fin=time.perf_counter()
    camino_l = [int(nodo) for nodo in CH_local[0]]
    print(f"{i:<4} | {alpha_val:<5.1f} | {'Local':<10} | {fin-ini:<10.4f} | {CH_local[1]:<8.2f} | {camino_l}")

    # UNIFORME
    ini=time.perf_counter()
    CH_uniforme=coloniaHormigas(G,num_hormigas=20,tmax=400,alpha=alpha_val,beta=2.5,rho=0.1,Q=10,metodo_actualizacion="uniforme")
    fin=time.perf_counter()
    camino_u = [int(nodo) for nodo in CH_uniforme[0]]
    print(f"{i:<4} | {alpha_val:<5.1f} | {'Uniforme':<10} | {fin-ini:<10.4f} | {CH_uniforme[1]:<8.2f} | {camino_u}")
    
    print("-" * 110)