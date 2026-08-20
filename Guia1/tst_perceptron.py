import numpy as np

def tst_perceptron(datos_tst,w):
    M=len(datos_tst[:,0])# cantidad de patrones para el test
    N=len(datos_tst[0,:])#cantidad de entradas para el test

    entradas_tst= -1*np.ones((M,N))
    entradas_tst[:,1:N]=datos_tst[:,0:N-1]

    salidas_tst=datos_tst[:,N-1]

    aciertos=0
    for i in range(M):
        #calculamos las salidas dejando los ultimos pesos sinapticos obtenidos
        y= np.sign(np.dot(w,entradas_tst[i,:]))

        if ((salidas_tst[i] - y) == 0):
            aciertos += 1

    #obtenemos el ration de aciertos (aciertos / cantidad de datos del test)
    ratio= aciertos/M
    return ratio
#jas,mnabsnmsdmns dmnsdb