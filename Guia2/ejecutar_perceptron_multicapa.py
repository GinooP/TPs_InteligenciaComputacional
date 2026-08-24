# Funcion para probar con datos un perceptron multicapa
import numpy as np

class Capa:
    def __init__(self, W, y, delta):
        self.W = W # Matriz de pesos
        self.y = y # Vector de salidas
        self.delta = delta # Vector de errores

def ejecutar_perceptron_multicapa():
    ...