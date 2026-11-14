##### parametros multicapa
---Constantes MLP
eta = 0.01
epocas = 500
tol = 0.95
capas = (64,32)

MLPClassifier(hidden_layer_sizes=capas,activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                         tol=tol, verbose=False,n_iter_no_change=100,learning_rate='adaptive')

##### parametros arbol 

DecisionTreeClassifier(criterion='gini',splitter='best')

##### parametros SVM 

SVC(kernel='rbf', C=2.0)

##### Resultados obtenidos sin duplicar ni invertir los datos
### Prueba 1
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.8296, 0.8384, 0.8224]
Media = 0.8301 | Varianza = 0.000043

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.84, 0.8204, 0.822]
Media = 0.8275 | Varianza = 0.000079

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.886, 0.8944, 0.8876]
Media = 0.8893 | Varianza = 0.000013

### Prueba 2
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.8336, 0.8372, 0.848]
Media = 0.8396 | Varianza = 0.000037

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.8404, 0.8304, 0.8252]
Media = 0.8320 | Varianza = 0.000040

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.886, 0.8944, 0.8876]
Media = 0.8893 | Varianza = 0.000013

### Prueba 3
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.8444, 0.8544, 0.8184]
Media = 0.8391 | Varianza = 0.000230

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.8352, 0.8208, 0.8284]
Media = 0.8281 | Varianza = 0.000035

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.886, 0.8944, 0.8876]
Media = 0.8893 | Varianza = 0.000013

##### Resultados obtenidos duplicando e invirtiendo los datos del train
# prueba 1
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.7944, 0.8374, 0.7918]
Media = 0.8079 | Varianza = 0.000437

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.762, 0.7656, 0.7726]
Media = 0.7667 | Varianza = 0.000019

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.777, 0.8164, 0.7766]
Media = 0.7900 | Varianza = 0.000349

# prueba 2
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.7974, 0.8372, 0.7936]
Media = 0.8094 | Varianza = 0.000389

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.761, 0.7654, 0.7736]
Media = 0.7667 | Varianza = 0.000027

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.777, 0.8164, 0.7766]
Media = 0.7900 | Varianza = 0.000349

# prueba 3

--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.7976, 0.8376, 0.7936]
Media = 0.8096 | Varianza = 0.000395

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.7624, 0.7706, 0.7726]
Media = 0.7685 | Varianza = 0.000019

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.777, 0.8164, 0.7766]
Media = 0.7900 | Varianza = 0.000349