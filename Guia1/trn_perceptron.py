import numpy as np

def trn_perceptron(datos, tasa_de_aprendizaje, maxit):

    N = len(datos[:,0]) # Cantidad de patrones (pruebas)
    M = len(datos[0,:]) # Cantidad de entradas + la del sesgo

    y_d = datos[:,-1] # Vector con salidas esperadas (la ultima columna)

    x = -1 * np.ones((N,M))
    x[:,1:M] = datos[:,0:-1]

    rng = np.random.default_rng()
    w = rng.random(M) - 0.5 # Setear los pesos entre [-0.5 0.5] del vector de pesos sináptico

    historial_w = [] # matriz con los valores que tomaron los pesos
    historial_w.append(np.copy(w))
    # Etapa de aprendizaje 
    for i in range(maxit):
        for j in range(N):
            y = np.sign(np.dot(w,x[j,:]))
            w = w + tasa_de_aprendizaje/2 * (y_d[j] - y)*x[j,:]
            historial_w.append(np.copy(w))

        # Etapa de ratio de aciertos
        aciertos = 0
        for j in range(N):
            y = np.sign(np.dot(w,x[j,:]))
            if (y_d[j] - y) == 0:
                aciertos += 1
        
        ratio = aciertos / N

    matriz_historial = np.array(historial_w)
    return w, ratio, matriz_historial