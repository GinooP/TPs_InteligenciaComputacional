import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB                              # Naive-Bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    # Analsis Discriminante Lineal 
from sklearn.neighbors import KNeighborsClassifier                      # K vecinos más cercanos
from sklearn.tree import DecisionTreeClassifier                         # Arbol de Decision
from sklearn.svm import SVC                                             # Maquina de Soporte Vectorial

digits = load_digits() #cargamos los datos del conjunto digits

X = digits.data
Y = digits.target

#hacemos una particion de los datos con train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

# Fold con 5 particiones
eta = 0.01
epocas = 500
tol = 1e-4

fk = KFold(n_splits=5)#lo dejamos con shufle en false

vector_aciertos = []
vector_aciertos_NB = []
vector_aciertos_LDA = []
vector_aciertos_KNN = []
vector_aciertos_AD = []
vector_aciertos_SVC = []
for i,(index_train,index_test) in enumerate(fk.split(X)):
    # print(f"Fold = {i}")

    X_train = X[index_train]
    Y_train = Y[index_train]

    X_test = X[index_test]
    Y_test = Y[index_test]

    # --- 1. Perceptrón Multicapa (MLP) ---
    modelo = MLPClassifier(hidden_layer_sizes=(100,80),activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                       random_state=42, tol=tol, verbose=False) #para que no muestre en consola todo el proceso
    modelo.fit(X_train,Y_train)
    vector_aciertos.append(modelo.score(X_test,Y_test))

    # --- 2. Naive-Bayes (NB) ---
    gnb = GaussianNB()
    gnb.fit(X_train,Y_train) # entrenamos el modelo
    vector_aciertos_NB.append(gnb.score(X_test,Y_test))
    
    # --- 3. Análisis Discriminante Lineal (LDA) ---
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, Y_train)
    vector_aciertos_LDA.append(clf.score(X_test,Y_test))

    # --- 4. K Vecinos Más Cercanos (KNN) ---
    clf_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
    clf_knn.fit(X_train, Y_train)
    vector_aciertos_KNN.append(clf_knn.score(X_test, Y_test))
    # 1 vecino -> Media = 0.9650 | Varianza = 0.000166
    # 5 vecinos -> Media = 0.9650 | Varianza = 0.000113
    # 10 vecinos -> Media = 0.9566 | Varianza = 0.000162
    # 100 vecinos -> Media = 0.8993 | Varianza = 0.000283
    # 1000 vecinos -> Media = 0.5275 | Varianza = 0.005013
    # 5 vecinos y pesos segun la distancia -> Media = 0.9649 | Varianza = 0.000100

    # --- 5. Arbol de decisión (AD) ---
    tree = DecisionTreeClassifier(criterion='gini',splitter='best')
    tree.fit(X_train,Y_train)
    vector_aciertos_AD.append(tree.score(X_test,Y_test))
    
    # --- 6. Máquina de soporte vectorial (SVC) ---
    clasificador_SVC = SVC(kernel='linear', C=1.0)
    clasificador_SVC.fit(X_train, Y_train)
    vector_aciertos_SVC.append(clasificador_SVC.score(X_test, Y_test))

# -------------------- IMPRESIÓN DE RESULTADOS --------------------
N = len(vector_aciertos)

print('\n--- Perceptrón Multicapa (MLP) ---')
media_mlp = np.sum(vector_aciertos)/N
varianza_mlp = np.sum((vector_aciertos - media_mlp)**2)/N
print(f"Ratios obtenidos = {vector_aciertos}")
print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

print('\n--- Naive-Bayes (NB) ---')
media_nb = np.sum(vector_aciertos_NB)/N
varianza_nb = np.sum((vector_aciertos_NB - media_nb)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_NB}")
print(f"Media = {media_nb:.4f} | Varianza = {varianza_nb:.6f}")

print('\n--- Análisis Discriminante Lineal (LDA) ---')
media_lda = np.sum(vector_aciertos_LDA)/N
varianza_lda = np.sum((vector_aciertos_LDA - media_lda)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_LDA}")
print(f"Media = {media_lda:.4f} | Varianza = {varianza_lda:.6f}")

print('\n--- K Vecinos Más Cercanos (KNN) ---')
media_knn = np.sum(vector_aciertos_KNN)/N
varianza_knn = np.sum((vector_aciertos_KNN - media_knn)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_KNN}")
print(f"Media = {media_knn:.4f} | Varianza = {varianza_knn:.6f}")


print('\n--- Árbol de Decisión (AD) ---')
media_ad = np.sum(vector_aciertos_AD)/N
varianza_ad = np.sum((vector_aciertos_AD - media_ad)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_AD}")
print(f"Media = {media_ad:.4f} | Varianza = {varianza_ad:.6f}")

print('\n--- Máquina de Soporte Vectorial (SVC) ---')
media_svm = np.sum(vector_aciertos_SVC)/N
varianza_svm = np.sum((vector_aciertos_SVC - media_svm)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_SVC}")
print(f"Media = {media_svm:.4f} | Varianza = {varianza_svm:.6f}")
