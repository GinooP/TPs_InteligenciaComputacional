import numpy as np
import math

class poblacion:
    def __init__(self,matriz_poblacion,fitnes):
        self.matriz_poblacion =matriz_poblacion
        self.fitnes = np.zeros((len(matriz_poblacion[:,1])))


def mutacion(individuo,cant_bits,prob_mutacion): #alteramos un bit al azar del individuo si corresponde mutarlo, si no lo dejamos como esta
    if(np.random.rand() < prob_mutacion):
        bit_mutante = np.random.randint(0,cant_bits)
        #print(f"el bit mutante es= {bit_mutante}")
        #print(f"indiviuo original= {individuo}")
        if(individuo[bit_mutante]==1):
            individuo[bit_mutante]=0
        else:
            individuo[bit_mutante]=1
        #print(f"individuo mutado= {individuo}")

def algoritmo_evolutivo(cant_individuos,cant_bits,func_fitnes,cant_progenitores,aptitud_requerida,prob_mutacion,inicializar,max_it,elitismo=False):

    #1 Inicializacion (podria agregar algun parametro para ajustar la probabilidad de unos o ceros)
    poblacion=inicializar(cant_individuos,cant_bits) #inicializamos la poblacion con valores al azar entre cero y uno
    print(poblacion)

    fitnes = func_fitnes(poblacion)
    fitnes=np.array(fitnes)

    mejores_fitnes=[]#aca vamos a guardar los mejores fitnes
    peores_fitnes=[]#aca vamos a guardar los peores fitnes

    #podriamos ordenar la poblacion segun sus fitnes
    indices = np.argsort(fitnes)[::-1] #argsort devuelve el vector de indices ya ordenados de poblacion fitnes con [::-1] invertimos el orden (orden decreciente en este caso)

    mejores_fitnes.append(fitnes[indices[0]])
    peores_fitnes.append(fitnes[indices[-1]])

    print(fitnes[indices])
    print(f"mejor fitnes: {fitnes[indices[0]]}")
    print(f"peor fitnes= {fitnes[indices[-1]]}")

    factor_elitismo=0
    if (elitismo):
        factor_elitismo=1

    num_generacion=0
    while(fitnes[indices[0]] < aptitud_requerida and num_generacion<max_it): #la mejor aptitud es el indice[0]
        num_generacion+=1
        print (f"generacion= {num_generacion}")
        #obtenemos los progenitores
        progenitores=[]
        cont=0
        decremento_ventana=int(cant_individuos/cant_progenitores)
        limite_ventana = cant_individuos + decremento_ventana #sumamos un decremento para restarlo abajo
        while(cont<cant_progenitores):
            #decrementamos la ventana
            limite_ventana -= decremento_ventana #vamos decrementando la ventana

            #de cada ventana sacamos un padre
            ind_padre=np.random.randint(0,limite_ventana)
            progenitores.append(poblacion[indices[ind_padre],:])#nos guardamos el padre

            cont +=1
        
        #hacemos el cruzamiento
        progenitores=np.array(progenitores)#para que compile los transformo en arrays de numpy
        cant_cruzamientos = round(cant_individuos/2)
        pos_repoblacion= factor_elitismo
        for i in range(cant_cruzamientos):#factor elitismo determina si dejamos el mas apto o no (si es 1 lo dejamos , si no lo remplazamos)
            #elegimos los pares de progenitores al azar

            ind_progenitor_1= np.random.randint(0,cant_progenitores)
            ind_progenitor_2=np.random.randint(0,cant_progenitores)
            while(ind_progenitor_1 == ind_progenitor_2): #para asegurarnos que no se repitan
                ind_progenitor_2= np.random.randint(0,cant_progenitores)

            #elegimos el bit de cruzamiento al azar
            ind_cruzamiento = np.random.randint(0,cant_bits)
            #obtemos los hijos
            hijo1=np.zeros(cant_bits)
            hijo2=np.zeros(cant_bits)

            
            hijo1[0:ind_cruzamiento] = progenitores[ind_progenitor_1, 0:ind_cruzamiento]
            hijo1[ind_cruzamiento:] = progenitores[ind_progenitor_2, ind_cruzamiento:]

            # print("##############################################################################################")
            # print(f"este es el progenitor 1 {progenitores[ind_progenitor_1,:]}")
            # print(f"este es el progenitor 2 {progenitores[ind_progenitor_2,:]}")
            # print(f"este es el hijo 1: {hijo1}")
            #determinamos si mutamos
            mutacion(hijo1,cant_bits,prob_mutacion)

            hijo2[0:ind_cruzamiento] = progenitores[ind_progenitor_2, 0:ind_cruzamiento]
            hijo2[ind_cruzamiento:] = progenitores[ind_progenitor_1, ind_cruzamiento:]

            mutacion(hijo2,cant_bits,prob_mutacion)


            poblacion[pos_repoblacion]= hijo1
            if (pos_repoblacion+1 < cant_individuos):
                poblacion[pos_repoblacion+1]= hijo2

            pos_repoblacion += 2

        #calculamos los fitnes de los nueva poblacion
        fitnes=func_fitnes(poblacion)
        fitnes=np.array(fitnes)
        #ordenamos la poblacion nuevamente (obteniendo sus indices en orden decreciente)
        indices= np.argsort(fitnes)[::-1]
        mejores_fitnes.append(fitnes[indices[0]])
        peores_fitnes.append(fitnes[indices[-1]])
        print("#########################################################################################")
        #print(fitnes[indices])
        print(f"mejor fitnes: {fitnes[indices[0]]}")
        print(f"peor fitnes= {fitnes[indices[-1]]}")
        print(f"cantidad de caracteristicas: {np.sum(poblacion[indices[0]])}")
        

    #si ya alcanzamos la aptitud requerida retornamos la mejor solucion y los historicos de mejores y peores fitnes

    return poblacion[indices[0],:],mejores_fitnes,peores_fitnes

        
        
        

