import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB                              # Naive-Bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis    # Analsis Discriminante Lineal 
from sklearn.neighbors import KNeighborsClassifier                      # K vecinos más cercanos
from sklearn.tree import DecisionTreeClassifier                         # Arbol de Decision
from sklearn.svm import SVC                                             # Maquina de Soporte Vectorial

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

wines = load_wine() #cargamos los datos del conjunto wine

X = wines.data
Y = wines.target

# Reducir de 13 dimensiones a 2 usando PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='plasma', edgecolor='k')

# plt.xlabel('Componente Principal 1')
# plt.ylabel('Componente Principal 2')
# plt.title('Dataset Wine comprimido a 2D usando PCA')
# plt.legend(handles=scatter.legend_elements()[0], labels=list(wines.target_names), title="Clases")

# plt.show()

#hacemos una particion de los datos con train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# Fold con 5 particiones
# eta = 0.0001
epocas = 10
tol = 1e-4

fk = KFold(n_splits=5, shuffle=True) #lo dejamos con shufle en false

vector_aciertos = []    # Para KFold
vector_aciertos_B = []  # Para Bagging
vector_aciertos_AB = [] # Para AdaBoost
for i,(index_train,index_test) in enumerate(fk.split(X)):
    # print(f"Fold = {i}")

    X_train = X[index_train]
    Y_train = Y[index_train]

    X_test = X[index_test]
    Y_test = Y[index_test]

    # --- 1. Perceptrón Multicapa (MLP) ---
    modelo = MLPClassifier(hidden_layer_sizes=(1000),activation='logistic',solver='sgd',max_iter=epocas,
                       random_state=42, tol=tol, verbose=False, n_iter_no_change=100, learning_rate='adaptive') #para que no muestre en consola todo el proceso
    modelo.fit(X_train,Y_train)
    vector_aciertos.append(modelo.score(X_test,Y_test))
    
    # neuronas = (100,100), eta = 0.0001, epocas = 20000 -> Media = 0.6746 | Varianza = 0.005943
    #   Ratios obtenidos = [0.6388888888888888, 0.5833333333333334, 0.7222222222222222, 0.8, 0.6285714285714286]
    # neuronas = (100), eta = 0.0001, epocas = 1000 -> Media = 0.6508 | Varianza = 0.004919
    #   Ratios obtenidos = [0.6944444444444444, 0.75, 0.6666666666666666, 0.5714285714285714, 0.5714285714285714]
    # neuronas = (10), eta = 0.0001, epocas = 1000 -> Media = 0.6517 | Varianza = 0.002046
    #   Ratios obtenidos = [0.6944444444444444, 0.6388888888888888, 0.6111111111111112, 0.6, 0.7142857142857143]
    # neuronas = (10), eta = 0.001, epocas = 1000 -> Media = 0.5783 | Varianza = 0.017457
    #   Ratios obtenidos = [0.6944444444444444, 0.3888888888888889, 0.7222222222222222, 0.45714285714285713, 0.6285714285714286]
    
    # neuronas = (2000), eta = 'adaptative', epocas = 2000 -> Media = 0.8871 | Varianza = 0.002968
    #   Ratios obtenidos = [0.9444444444444444, 0.8611111111111112, 0.9444444444444444, 0.8, 0.8857142857142857]
    # neuronas = (1000), eta = 'adaptative', epocas = 2000 -> Media = 0.8595 | Varianza = 0.001310
    #   Ratios obtenidos = [0.8611111111111112, 0.8611111111111112, 0.8611111111111112, 0.8, 0.9142857142857143]

    # --- 2. Bagging (B) ---
    clf_B = BaggingClassifier(estimator=None, n_estimators=10, max_samples=None, max_features=1.0, bootstrap=True, 
                        bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=42, verbose=0)
    # estimator = ['SVC()'|'LogisticRegression()'|'GaussianNB()'|'MLPClassifier(...)'|'KNeighborsClassifier()'|etc]
    clf_B.fit(X_train,Y_train)
    vector_aciertos_B.append(clf_B.score(X_test,Y_test))

    # estimator=None -> Media = 0.9606 | Varianza = 0.000810
    # estimator=SVC(kernel='linear', C=1.0) -> Media = 0.9552 | Varianza = 0.000501
    # estimator=DecisionTreeClassifier(criterion='gini',splitter='best') -> Media = 0.9549 | Varianza = 0.000200
    # estimator=KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto') -> Media = 0.7192 | Varianza = 0.004564
    # estimator=LinearDiscriminantAnalysis() -> Media = 0.9832 | Varianza = 0.000497
    # estimator=GaussianNB() -> Media = 0.9665 | Varianza = 0.000739
    
    # --- 3. AdaBoost (AB) ---
    clf_AB = AdaBoostClassifier(estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)
    # learning_rate: peso que se le asigna a los clasificadores en cada iteracion
    # n_estimators: numero máximo de estimadores
    clf_AB.fit(X_train, Y_train)
    vector_aciertos_AB.append(clf_AB.score(X_test,Y_test))

    # estimator=None -> Media = 0.9330 | Varianza = 0.001704
    # estimator=SVC(kernel='linear', C=1.0) -> Media = 0.9379 | Varianza = 0.001111
    # estimator=DecisionTreeClassifier(criterion='gini',splitter='best') -> Media = 0.9048 | Varianza = 0.002039
    # estimator=KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto') -> Error
    # estimator=LinearDiscriminantAnalysis() -> Error
    # estimator=GaussianNB() -> Media = 0.9717 | Varianza = 0.000327
    
# -------------------- IMPRESIÓN DE RESULTADOS --------------------
N = len(vector_aciertos)

print('\n--- Perceptrón Multicapa (MLP) ---')
media_mlp = np.sum(vector_aciertos)/N
varianza_mlp = np.sum((vector_aciertos - media_mlp)**2)/N
print(f"Ratios obtenidos = {vector_aciertos}")
print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

print('\n--- Bagging (B) ---')
media_nb = np.sum(vector_aciertos_B)/N
varianza_nb = np.sum((vector_aciertos_B - media_nb)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_B}")
print(f"Media = {media_nb:.4f} | Varianza = {varianza_nb:.6f}")

print('\n--- AdaBost (AB) ---')
media_lda = np.sum(vector_aciertos_AB)/N
varianza_lda = np.sum((vector_aciertos_AB - media_lda)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_AB}")
print(f"Media = {media_lda:.4f} | Varianza = {varianza_lda:.6f}")


