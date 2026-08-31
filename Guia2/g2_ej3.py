import numpy as np
import perceptron_multicapa as mp


# Etapa de Entrenamiento
patrones_trn = np.loadtxt('Guia2/iris81_trn.csv', delimiter=',', skiprows=0)
patrones_tst = np.loadtxt('Guia2/iris81_tst.csv', delimiter=',', skiprows=0)

epocas = 200
etas_a_probar = [0.01, 0.05, 0.1]
neuronas_ocultas_a_probar = [1, 2, 3]
tol = 100 # Las tolerancias que definiste para cada caso

for idx, capa_oculta in enumerate(neuronas_ocultas_a_probar):
    print(f'\n-----------------------PRUEBA CON {capa_oculta} NEURONAS OCULTAS-----------------------')
    
    capas = [4, capa_oculta, 3]
    
    errores_cuad_por_eta = []
    errores_clas_por_eta = []
    
    for eta in etas_a_probar:
        print(f"Entrenando con eta = {eta}...")
        vector_capas, ratios, errores, errores_de_clasificacion = mp.generar_perceptron_multicapa(capas, patrones_trn, eta, epocas, tol, True)
        ratio, error = mp.tst_perceptron_multicapa(patrones_tst, vector_capas, capas, True)
        
        # Guardamos el historial de esta corrida para graficarlo después
        errores_cuad_por_eta.append(errores)
        errores_clas_por_eta.append(errores_de_clasificacion)
        
    # Llamamos a la función para graficar los resultados de esta configuración
    mp.graficar_resultados(etas_a_probar, errores_cuad_por_eta, errores_clas_por_eta, capa_oculta)