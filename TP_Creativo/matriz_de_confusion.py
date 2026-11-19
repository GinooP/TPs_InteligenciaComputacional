import numpy as np

def obtener_matriz_de_confusion(Y_correctas,Y_predichas):
    matriz_confusion=np.zeros((2,2))

    n= len(Y_correctas)

    for i in range(n):
        matriz_confusion[int(Y_correctas[i]),int(Y_predichas[i])] += 1 #incrementamos los valores

    return matriz_confusion