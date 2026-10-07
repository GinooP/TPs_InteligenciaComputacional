import numpy as np
import computacion_evolutiva as evo
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

def f_fitness(poblacion):
    patrones_train= np.loadtxt('Guia 6/leukemia_train.csv', delimiter=',', skiprows=0)
    patrones_test = np.loadtxt('Guia 6/leukemia_test.csv', delimiter=',', skiprows=0)

    N= len(poblacion[:,0])
    x_train=patrones_train[:,:-1]
    y_train= patrones_train[:,-1]

    x_test=patrones_test[:,:-1]
    y_test= patrones_test[:,-1]
    
    fitness=np.zeros((N))

    for i in range(N):
        mascara = poblacion[i,:]
        x_train_i=x_train[:,mascara == 1]

        eta=0.01
        epocas=200
        tol=1e-4
        modelo=SVC(kernel='linear', C=1.0)
        modelo.fit(x_train_i, y_train)
        
        x_test_i=x_test[:,mascara==1]

        fitness[i]= modelo.score(x_test_i,y_test)

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

aptitud_requerida=0.99#obtenido por inspeccion visual en geogebra (podria estimarlo con biseccion en octave)

solucion,mejores_fitnes,peores_fitnes = evo.algoritmo_evolutivo(cant_individuos,cant_bits,f_fitness,cant_progenitores,aptitud_requerida,prob_mutacion,inicializar=inicializar,elitismo=True)

        