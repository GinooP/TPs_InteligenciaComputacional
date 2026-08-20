import numpy as np
import matplotlib.pyplot as plt
import time
import trn_perceptron as trn
import tst_perceptron as tst

datos_OR_trn=np.loadtxt("Guia1/OR_trn.csv",delimiter=',',skiprows=0)
datos_XOR_trn=np.loadtxt("Guia1/XOR_trn.csv",delimiter=',',skiprows=0)

#obtenemos los pesos sinapticos y el historial de pesos sinapticos

itmax=100
tasa_de_aprendizaje=0.1

w_OR, ratio_trn_OR, historial_w_OR = trn.trn_perceptron(datos_OR_trn,tasa_de_aprendizaje,itmax)
w_XOR, ratio_trn_XOR, historial_w_XOR = trn.trn_perceptron(datos_XOR_trn,tasa_de_aprendizaje,itmax)

# testeamos con nuevos patrones

datos_OR_tst=np.loadtxt("Guia1/OR_tst.csv",delimiter=',',skiprows=0)
datos_XOR_tst=np.loadtxt("Guia1/XOR_tst.csv",delimiter=',',skiprows=0)

# ratio_tst_OR= tst.tst_perceptron(datos_OR_tst,w_OR)
# ratio_tst_XOR= tst.tst_perceptron(datos_XOR_tst,w_XOR)

######### graficamos OR ##########################
cant_pesos = len(historial_w_XOR[:,0])
yd_XOR=datos_XOR_tst[:,-1]# salida esperada XOR
yd_OR=datos_OR_tst[:,-1]# salida esperada or

x_recta1 = np.linspace(min(datos_XOR_tst[:,0]) - 0.5, max(datos_XOR_tst[:,0]) + 0.5, 100)
x_recta2 = np.linspace(min(datos_OR_tst[:,0]) - 0.5, max(datos_OR_tst[:,0]) + 0.5, 100)

plt.ion()
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

for i in range(cant_pesos):
    # Calculo de las rectas
    y_recta1 = historial_w_XOR[i,0]/historial_w_XOR[i,2] - (historial_w_XOR[i,1]/historial_w_XOR[i,2])*x_recta1
    y_recta2 = historial_w_OR[i,0]/historial_w_OR[i,2] - (historial_w_OR[i,1]/historial_w_OR[i,2])*x_recta2

    ax.clear()
    ax2.clear()

    ax.scatter(datos_XOR_tst[:,0], datos_XOR_tst[:,1], c=yd_XOR, cmap='bwr')
    ax2.scatter(datos_OR_tst[:,0], datos_OR_tst[:,1], c=yd_OR, cmap='bwr')
    
    ax.plot(x_recta1,y_recta1,color='green')
    ax2.plot(x_recta2, y_recta2, color='green')

    ax.set_xlim(-1.5, 1.5) 
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("XOR")
    ax2.set_xlim(-1.5, 1.5) 
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title("OR")
    plt.pause(0.3)

plt.ioff()

plt.show()
