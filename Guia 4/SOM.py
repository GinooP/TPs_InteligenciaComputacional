import numpy as np

def buscar_indice_activacion(matriz_neuronas,dim_matriz_neuronas,patron):

    #busca la menor distancia
    #primero busca en la primera fila
    #print(f"matriz_neuironas= {matriz_neuronas[0,0]}")
    #print(f"patron: {patron}")
    menor= sum((patron - matriz_neuronas[0,0])**2)
    #print(f"este es el menor: {menor}")
    pos_act=(0,0)
    for j  in range(1,dim_matriz_neuronas[1]):
        dist = sum((patron - matriz_neuronas[0,j])**2)#distancia euclidea (sin la raiz cuadrada)

        if(dist < menor):
            menor=dist
            pos_act=(0,j)

    #luego busca en el resto de la matriz
    for i in range(dim_matriz_neuronas[0]):
        for j in range(dim_matriz_neuronas[1]):
            dist = sum((patron - matriz_neuronas[i,j])**2)#distancia euclidea (sin la raiz cuadrada)

            if(dist < menor):
                menor=dist
                pos_act=(i,j)

    return pos_act

def som(patrones,dim_matriz_neuronas,epocas,eta,vecindad,cuadrado):
    #cuadrado=indica si la vencidad tiene forma cuadrada o de rombo
    #vecindad = determina hasta cuantos vecinos tomamos (con forma de rombo o cuadrado dependiendo del bool
    #dim_matriz_neuronas: dimensiones de la matriz de neuronas (filas,columnas)

    ###################### inicilaizacion de los pesos ###################################
    N=len(patrones[0,:])#obtnemos la cantidad de columanas para determinar el tamaño del vector de pesos
    cant_patrones=len(patrones[:,1])#obtenemos la cantidad de patrones

    rng = np.random.default_rng()
    matriz_neuronas= rng.random(size=(dim_matriz_neuronas[0],dim_matriz_neuronas[1],N))

    ##################################
    for n in range(epocas):
        
        for k in range(cant_patrones):#recorremos los patrones
            #buscamos la neurona que debe activarse
            pos_act=buscar_indice_activacion(matriz_neuronas,dim_matriz_neuronas,patrones[k]) # funciona pero lo podria cambiar por una operacion vectorial ahora que solucione las dimensiones
            #actualizamos los pesos

            if(cuadrado==True):
                ###### con esto genero un cuadrado ##########################
                #obtenemos desplazamientos hacia la izquierda, derecha ,arriba y abajo para no salirnos del margen
                #desp_arriba
                if(pos_act[0] - vecindad < 0):
                    desp_arriba=pos_act[0] #si i_act=0 no me puedo desplazar para arriba. si i_act=1 solo me puedo desplazar un lugar para arriba y así
                else:
                    desp_arriba=vecindad

                #desp_abajo
                if(pos_act[0] + vecindad >= dim_matriz_neuronas[0]):
                    desp_abajo=dim_matriz_neuronas[0]-1 - pos_act[0]
                else:
                    desp_abajo=vecindad

                #desp_izq
                if(pos_act[1] - vecindad < 0):
                    desp_izq=pos_act[1]#
                else:
                    desp_izq=vecindad

                #desp_derecha
                if(pos_act[1] + vecindad >= dim_matriz_neuronas[1]):
                    desp_der= dim_matriz_neuronas[1]-1 - pos_act[1]
                else:
                    desp_der= vecindad

                delta_pesos=eta*(patrones[k] - matriz_neuronas[pos_act[0]-desp_arriba:pos_act[0]+desp_abajo, pos_act[1]-desp_izq:pos_act[1]+desp_der])
                matriz_neuronas[pos_act[0]-desp_arriba:pos_act[0]+desp_abajo, pos_act[1]-desp_izq:pos_act[1]+desp_der] += delta_pesos
            else:
                #con esto formamos un rombo
                
                #determinamos los limites verticales
                #desp_arriba
                if(pos_act[0] - vecindad < 0):
                    desp_arriba=pos_act[0] #si i_act=0 no me puedo desplazar para arriba. si i_act=1 solo me puedo desplazar un lugar para arriba y así
                else:
                    desp_arriba=vecindad

                #desp_abajo
                if(pos_act[0] + vecindad >= dim_matriz_neuronas[0]):
                    desp_abajo=dim_matriz_neuronas[0]-1 - pos_act[0]
                else:
                    desp_abajo=vecindad
                #determinamos los limites horizontales
                #desp_izq
                if(pos_act[1] - vecindad < 0):
                    desp_izq=pos_act[1]#
                else:
                    desp_izq=vecindad

                #desp_derecha
                if(pos_act[1] + vecindad >= dim_matriz_neuronas[1]):
                    desp_der= dim_matriz_neuronas[1]-1 - pos_act[1]
                else:
                    desp_der= vecindad

                matriz_neuronas[pos_act[0],pos_act[1]-desp_izq:pos_act[1]+desp_der] += eta*(patrones[k] - matriz_neuronas[pos_act[0],pos_act[1]-desp_izq:pos_act[1]+desp_der])
                #cada barrido horizontal se reduce en una unidad por izquierda y otra por derecha
                for i in range(1,desp_arriba):
                    matriz_neuronas[pos_act[0]-i,pos_act[1]-(desp_izq-i):pos_act[1]+(desp_der-i)] += eta*(patrones[k] - matriz_neuronas[pos_act[0]-i,pos_act[1]-(desp_izq-i):pos_act[1]+(desp_der-i)])

                for i in range(1,desp_abajo):
                    matriz_neuronas[pos_act[0]+i,pos_act[1]-(desp_izq-i):pos_act[1]+(desp_der-i)] += eta*(patrones[k] - matriz_neuronas[pos_act[0]+i,pos_act[1]-(desp_izq-i):pos_act[1]+(desp_der-i)])

    #deberiamos poder hacer que tanto la vecindad como el eta se reduzcan en forma lineal...
    #... o tomar unos valores para la primera etapa , otros para la segundo y otro para la ultima
                    
                    
                    

                

                    