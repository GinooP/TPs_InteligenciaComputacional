import numpy as np
import matplotlib.pyplot as plt

datos_trn= np.loadtxt("Guia1 auxiliar/OR_trn.csv",delimiter=',',skiprows=0) #leemos los datos del archivo de texto
#print(datos_trn)

M=len(datos_trn[:,0]) #obtenemos la cantidad de patrones, osea datos
N=len(datos_trn[0,:]) #obtenemos la cantidad de entradas (las columnas)

#agregamos la entrada del bias (la del x0)
entradas_trn= -1*np.ones((M,N)) 
entradas_trn[:,1:N]= datos_trn[:,0:N-1] #agregamos todas las entradas sin poner las salidas (la ultima fila)

#sacamos las salidas(no hace falta pero bueeeenooo)
salidas_trn=datos_trn[:,N-1]

#inicializamos los pesos de manera aleatoria entre -0.5 y 0.5

rng= np.random.default_rng()
w= rng.random(N) - 0.5 #creamos un vector de pesos sinapticos de tamaño igual a la cantidad de las entradas

#etapa de aprendisaje
coef_aprendisaje = 0.2 #usar valores bajos
maxit=200 #cantidad maxima de iteraciones
it_utilizadas=0
for it in range(maxit):
    for i in range(M):
        #hacemos el producto punto entre los pesos sinapticos y la salida
        #al producto lo metemos como entrada en la funcion de activacion (en este caso usamos la funcion signo)
        y = np.sign(np.dot(w,entradas_trn[i,:])) #calculamos la salida
    
        #ahora actualizamos los pesos usando la formula
        w = w + 0.5*coef_aprendisaje*(salidas_trn[i] - y)*entradas_trn[i,:]

    #etapa de verificacion
    aciertos=0
    for i in range(M):
        y=np.sign(np.dot(w,entradas_trn[i,:]))
        
        if ((salidas_trn[i] - y) == 0):
            aciertos += 1
    ratio_trn=aciertos/M
    print(f"el ratio de practica obtenido en la iteracion {it} es {ratio_trn*100}")
    if (ratio_trn == 1):
        it_utilizadas=it
        break

if(it_utilizadas!=0):
    print(f"se requirieron {it_utilizadas} iteraciones para el aprendizaje")
else:
    print(f"se requierieron {maxit} iteraciones para el aprendizaje")



#ahora hacemos las pruebas
datos_tst= np.loadtxt("Guia1 auxiliar/OR_tst.csv",delimiter=',',skiprows=0) #leemos los datos del test

M=len(datos_tst[:,0])# cantidad de patrones para el test
N=len(datos_tst[0,:])#cantidad de entradas para el test

entradas_tst= -1*np.ones((M,N))
entradas_tst[:,1:N]=datos_tst[:,0:N-1]

salidas_tst=datos_tst[:,N-1]

aciertos=0
for i in range(M):
    #calculamos las salidas dejando los ultimos pesos sinapticos obtenidos
    yd= np.sign(np.dot(w,entradas_tst[i,:]))

    if ((salidas_tst[i] - yd) == 0):
        aciertos += 1

#obtenemos el ration de aciertos (aciertos / cantidad de datos del test)
ratio= aciertos/M
print(f"El ratio de aciertos del test es de: {ratio*100} %")

########## Graficamos ###########################

x2_frontera = w[0]/w[2] - (w[1]/w[2])*entradas_tst[:,1]
plt.scatter(entradas_tst[:,2],entradas_tst[:,1])
plt.plot(x2_frontera,entradas_tst[:,1],color='red')
plt.show()