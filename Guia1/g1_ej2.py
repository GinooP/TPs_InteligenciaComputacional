import numpy as np
import matplotlib.pyplot as plt
import time
import trn_perceptron as trn
import tst_perceptron as tst

datos_OR_trn=np.loadtxt("Guia1 auxiliar/OR_trn.csv",delimiter=',',skiprows=0)
datos_XOR_trn=np.loadtxt("Guia1 auxiliar/XOR_trn.csv",delimiter=',',skiprows=0)

#obtenemos los pesos sinapticos y el historial de pesos sinapticos

itmax=100
tasa_de_aprendizaje=0.2

w_OR, ratio_trn_OR, historial_w_OR = trn.trn_perceptron(datos_OR_trn,tasa_de_aprendizaje,itmax)
w_XOR, ratio_trn_XOR, historial_w_XOR = trn.trn_perceptron(datos_XOR_trn,tasa_de_aprendizaje,itmax)

#testeamos con nuevos patrones

datos_OR_tst=np.loadtxt("Guia1 auxiliar/OR_tst.csv",delimiter=',',skiprows=0)
datos_XOR_tst=np.loadtxt("Guia1 auxiliar/XOR_tst.csv",delimiter=',',skiprows=0)

# ratio_tst_OR= tst.tst_perceptron(datos_OR_tst,w_OR)
# ratio_tst_XOR= tst.tst_perceptron(datos_XOR_tst,w_XOR)

#### Graficamos las distintas fronteras del XOR
M=len(datos_XOR_tst[0,:])#cantidad de entradas para el test
N=len(datos_XOR_tst[0,:])#cantidad de entradas para el test





cant_pesos = len(historial_w_XOR[:,0])
yd_XOR=datos_XOR_tst[:,-1]# salida esperada XOR
plt.figure(1)
plt.ion()
fig, ax = plt.subplots()

for i in range(cant_pesos):
    x2_frontera = historial_w_XOR[i,0]/historial_w_XOR[i,2] - (historial_w_XOR[i,1]/historial_w_XOR[i,2])*datos_XOR_tst[:,0]
    ax.clear()
    ax.scatter(datos_XOR_tst[:,1],datos_XOR_tst[:,0],c=yd_XOR,cmap='bwr')
    ax.plot(x2_frontera,datos_XOR_tst[:,0],color='green')
    ax.set_xlim(-1.5, 1.5) 
    ax.set_ylim(-1.5, 1.5)
    plt.pause(0.3)

plt.ioff()

######### graficamos OR ##########################
plt.figure(2)
cant_pesos = len(historial_w_OR[:,0])
yd_OR=datos_OR_tst[:,-1]# salida esperada or
plt.ion()
fig2, ax2 = plt.subplots()

for i in range(cant_pesos):
    x2_frontera = historial_w_OR[i,0]/historial_w_OR[i,2] - (historial_w_OR[i,1]/historial_w_OR[i,2])*datos_OR_tst[:,0]
    ax2.clear()
    ax2.scatter(datos_OR_tst[:,1],datos_OR_tst[:,0],c=yd_OR,cmap='bwr')
    ax2.plot(x2_frontera,datos_OR_tst[:,0],color='green')
    ax2.set_xlim(-1.5, 1.5) 
    ax2.set_ylim(-1.5, 1.5)
    plt.pause(0.3)

plt.ioff()


plt.show()
