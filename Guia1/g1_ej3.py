import numpy as np
import matplotlib.pyplot as plt
import time
import trn_perceptron as trn
import tst_perceptron as tst

datos_OR_50_trn=np.loadtxt("Guia1/OR_50_trn.csv",delimiter=',',skiprows=0)
datos_OR_90_trn=np.loadtxt("Guia1/OR_90_trn.csv",delimiter=',',skiprows=0)

#obtenemos los pesos sinapticos y el historial de pesos sinapticos

itmax=1000
tasa_de_aprendizaje=0.2

w_OR_50, ratio_trn_OR_50, historial_w_OR_50 = trn.trn_perceptron(datos_OR_50_trn,tasa_de_aprendizaje,itmax)
w_OR_90, ratio_trn_OR_90, historial_w_OR_90 = trn.trn_perceptron(datos_OR_90_trn,tasa_de_aprendizaje,itmax)

#testeamos con nuevos patrones

datos_OR_50_tst=np.loadtxt("Guia1/OR_50_tst.csv",delimiter=',',skiprows=0)
datos_OR_90_tst=np.loadtxt("Guia1/OR_90_tst.csv",delimiter=',',skiprows=0)

ratio_tst_OR_50=tst.tst_perceptron(datos_OR_50_tst,w_OR_50)
ratio_tst_OR_90=tst.tst_perceptron(datos_OR_90_tst,w_OR_90)

print(f"el ratio de aciertos del OR 50 es: {ratio_tst_OR_50*100}%")
print(f"El ratio de aciertos del OR 90  es: {ratio_tst_OR_90*100}%")

############### Graficamos para el OR 90 ############################
yd_OR_90=datos_OR_90_tst[:,-1]


cant_pesos = len(historial_w_OR_90[:,0])
print(f"cantidad de pesos del OR 90 es: {cant_pesos}")

plt.figure(1)
x2_frontera = w_OR_90[0]/w_OR_90[2] - (w_OR_90[1]/w_OR_90[2])*datos_OR_90_tst[:,0]
plt.scatter(datos_OR_90_tst[:,1],datos_OR_90_tst[:,0],c=yd_OR_90,cmap='bwr')
plt.plot(x2_frontera,datos_OR_90_tst[:,0],color='green')
plt.xlim(-1.5, 1.5) 
plt.ylim(-1.5, 1.5)
plt.title("OR 90")
plt.grid(1)

# plt.ion()
# fig, ax = plt.subplots()

# for i in range(cant_pesos):
#     x2_frontera = historial_w_OR_90[i,0]/historial_w_OR_90[i,2] - (historial_w_OR_90[i,1]/historial_w_OR_90[i,2])*datos_OR_90_tst[:,0]
#     ax.clear()
#     ax.scatter(datos_OR_90_tst[:,1],datos_OR_90_tst[:,0])
#     ax.plot(x2_frontera,datos_OR_90_tst[:,0],color='red')
#     ax.set_xlim(-1.5, 1.5) 
#     ax.set_ylim(-1.5, 1.5)
#     ax.set_title("OR 90")
#     plt.pause(0.1)

# plt.ioff()



############# Graficamos para el OR_50 ################################

cant_pesos = len(historial_w_OR_50[:,0])
print(f"cantidad de pesos del OR 50 es: {cant_pesos}")

yd_OR_50=datos_OR_50_tst[:,-1]# salida esperada del OR 90 test

plt.figure(2)
x2_frontera = w_OR_50[0]/w_OR_50[2] - (w_OR_50[1]/w_OR_50[2])*datos_OR_50_tst[:,0]
plt.scatter(datos_OR_50_tst[:,1],datos_OR_50_tst[:,0],c=yd_OR_50,cmap='bwr')
plt.plot(x2_frontera,datos_OR_50_tst[:,0],color='green')
plt.xlim(-1.5, 1.5) 
plt.ylim(-1.5, 1.5)
plt.title("OR 50")
plt.grid(1)
# plt.ion()
# fig2, ax2 = plt.subplots()

# for i in range(cant_pesos):
#     x2_frontera = historial_w_OR_50[i,0]/historial_w_OR_50[i,2] - (historial_w_OR_50[i,1]/historial_w_OR_50[i,2])*datos_OR_50_tst[:,0]
#     ax2.clear()
#     ax2.scatter(datos_OR_50_tst[:,1],datos_OR_50_tst[:,0])
#     ax2.plot(x2_frontera,datos_OR_50_tst[:,0],color='red')
#     ax2.set_xlim(-1.5, 1.5) 
#     ax2.set_ylim(-1.5, 1.5)
#     ax2.set_title("OR 50")
#     plt.pause(0.1)

# plt.ioff()
plt.show()
