import numpy as np
import computacion_evolutiva as evo
from sklearn.model_selection import KFold
from sklearn.svm import SVC

def f_fitness(poblacion):
    patrones_train= np.loadtxt('Guia 6/leukemia_train.csv', delimiter=',', skiprows=0)

    N= len(poblacion[:,0])
    X=patrones_train[:,:-1]
    Y= patrones_train[:,-1]

    fitness=np.zeros((N))

    k=4#cantidad de folds
    fk= KFold(n_splits=4)#lo dejamos con shufle en false

    for i in range(N): #recorremos los miembros de la poblacion


        mascara = poblacion[i,:]
        X_i=X[:,mascara == 1]

        vector_ratio_aciertos=np.zeros(k)
        for j,(index_train,index_test) in enumerate(fk.split(X_i)):
            

            X_train = X[index_train]
            Y_train = Y[index_train]

            X_test = X[index_test]
            Y_test = Y[index_test]

            modelo=SVC(kernel='linear', C=1.0)
            modelo.fit(X_train, Y_train)

            vector_ratio_aciertos[j]=modelo.score(X_test,Y_test)

        fitness[i]=sum(vector_ratio_aciertos)/k

    return fitness

def inicializar(cant_individuos,cant_bits):
    opciones = [0, 1]
    pesos = [0.99, 0.01]

    poblacion=np.random.choice(opciones,size=(cant_individuos,cant_bits),p=pesos)

    return poblacion


cant_individuos=16
cant_bits=7129
cant_progenitores=5
prob_mutacion=0.2#probabilidad de mutacion
elitismo=True
max_it=100

aptitud_requerida=0.94#obtenido por inspeccion visual en geogebra (podria estimarlo con biseccion en octave)

solucion,mejores_fitnes,peores_fitnes = evo.algoritmo_evolutivo(cant_individuos,cant_bits,f_fitness,cant_progenitores,aptitud_requerida,prob_mutacion,inicializar=inicializar,max_it=max_it,elitismo=True)

#entrenamos el modelo
patrones_train= np.loadtxt('Guia 6/leukemia_train.csv', delimiter=',', skiprows=0)
patrones_test= np.loadtxt('Guia 6/leukemia_test.csv', delimiter=',', skiprows=0)

X_train=patrones_train[:,:-1]
Y_train= patrones_train[:,-1]

X_test=patrones_test[:,:-1]
Y_test=patrones_test[:,-1]

X_train=X_train[:,solucion==1]

X_test=X_test[:,solucion==1]

modelo=SVC(kernel='linear', C=1.0)
modelo.fit(X_train, Y_train)

ratio_aciertos=modelo.score(X_test,Y_test)
print(f"el ratio de aciertos obtenido es= {ratio_aciertos}")


        