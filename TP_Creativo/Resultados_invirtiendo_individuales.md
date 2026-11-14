# constantes de los clasificadores

-----MLP
eta = 0.05
epocas = 1000
tol = 0.95
capas=(64,32)

modelo = MLPClassifier(hidden_layer_sizes=capas,activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas
                           , tol=tol, verbose=False) #para que no muestre en consola todo el proceso

------- arbol de desicion
tree = DecisionTreeClassifier(criterion='gini',splitter='best')

------- SVM
SVC(kernel='rbf', C=1.0)

### pruebas sin hacer la inversion y concatenacion

-------- prueba 1 -------------
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.87616, 0.87792, 0.87336]
Media = 0.8758 | Varianza = 0.000004

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.93408, 0.93464, 0.93448]
Media = 0.9344 | Varianza = 0.000000

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.89456, 0.89248, 0.8864]
Media = 0.8911 | Varianza = 0.000012

-------- prueba 2 -------------
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.87952, 0.87904, 0.86536]
Media = 0.8746 | Varianza = 0.000043

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.93456, 0.93568, 0.93624]
Media = 0.9355 | Varianza = 0.000000

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.89456, 0.89248, 0.8864]
Media = 0.8911 | Varianza = 0.000012

-------- prueba 3 -------------
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.86504, 0.87632, 0.85888]
Media = 0.8667 | Varianza = 0.000052

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.936, 0.93296, 0.93568]
Media = 0.9349 | Varianza = 0.000002

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.89456, 0.89248, 0.8864]
Media = 0.8911 | Varianza = 0.000012

### pruebas haciendo la inversion y concatenacion

-------- prueba 1 -------------
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.89028, 0.88384, 0.88104]
Media = 0.8851 | Varianza = 0.000015

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.90708, 0.92176, 0.9102]
Media = 0.9130 | Varianza = 0.000040

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.87936, 0.89032, 0.87772]
Media = 0.8825 | Varianza = 0.000031
-------- prueba 2 -------------
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.8828, 0.88688, 0.88284]
Media = 0.8842 | Varianza = 0.000004

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.90908, 0.92204, 0.91012]
Media = 0.9137 | Varianza = 0.000035

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.87936, 0.89032, 0.87772]
Media = 0.8825 | Varianza = 0.000031
-------- prueba 3 -------------
--- Perceptrón Multicapa (MLP) ---
Ratios obtenidos = [0.88436, 0.88532, 0.88076]
Media = 0.8835 | Varianza = 0.000004

--- Arbol de Decisión (AD) ---
Ratios obtenidos = [0.9076, 0.92116, 0.90996]
Media = 0.9129 | Varianza = 0.000035

--- Máquina de soporte vectorial (SVC) ---
Ratios obtenidos = [0.87936, 0.89032, 0.87772]
Media = 0.8825 | Varianza = 0.000031