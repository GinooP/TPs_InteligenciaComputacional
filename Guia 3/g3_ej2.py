import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier

def calcular_media_varianza(vector):
    N=len(vector)
    media= np.sum(vector)/N
    varianza= np.sum((vector - media)**2)/N
    return media,varianza

print("####################### Utilizando KFold con 5 particiones ###################################")
digits = load_digits() #cargamos los datos del conjunto digits

X= digits.data
Y= digits.target

eta =0.01
epocas=500
tol=1e-4

fk= KFold(n_splits=5)#lo dejamos con shufle en false

vector_aciertos_MLPC=np.array([])
vector_aciertos_NBayes=np.array([])
vector_aciertos_Tree=np.array([])
vector_aciertos_KNN=np.array([])
vector_aciertos_LDA=np.array([])
for i,(index_train,index_test) in enumerate(fk.split(X)):
    print(f"Fold= {i}")

    X_train = X[index_train]
    Y_train = Y[index_train]

    X_test = X[index_test]
    Y_test = Y[index_test]

    ###### Perceptron multicapa #################################
    #definimos el modelo dentro del for para entrenar el mismo modelo con cada fold
    modelo = MLPClassifier(hidden_layer_sizes=(100,80),activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                       random_state=42, tol=tol, verbose=False) #para que no muestre en consola todo el proceso

    #nota: una optimizacion posible es hacer modelo.fit(X[index_train],Y[index_train]) y lo mismo en test para no ocupar tanta memoria
    modelo.fit(X_train,Y_train)

    ratio_aciertos= modelo.score(X_test,Y_test)
    vector_aciertos_MLPC =np.append(vector_aciertos_MLPC,ratio_aciertos)

    ################# Naive Bayes ##############################################
    gnb=GaussianNB()

    gnb.fit(X_train,Y_train)#entrenamos el modelo

    ratio_aciertos_NBayes=gnb.score(X_test,Y_test)

    vector_aciertos_NBayes=np.append(vector_aciertos_NBayes,ratio_aciertos_NBayes)

    ############### Arbol de Decición ########################
    tree=DecisionTreeClassifier(criterion='gini',splitter='best')

    tree.fit(X_train,Y_train)

    ratio_aciertos_tree=tree.score(X_test,Y_test)

    vector_aciertos_Tree=np.append(vector_aciertos_Tree,ratio_aciertos_tree)

    # --- 3. Análisis Discriminante Lineal (LDA) ---
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, Y_train)
    vector_aciertos_LDA=np.append(vector_aciertos_LDA,clf.score(X_test,Y_test))

    # --- 4. K Vecinos Más Cercanos (KNN) ---
    clf_knn = KNeighborsClassifier(n_neighbors=5)
    clf_knn.fit(X_train, Y_train)
    vector_aciertos_KNN=np.append(vector_aciertos_KNN,clf_knn.score(X_test, Y_test))
    # 1 vecino -> Media = 0.9650 | Varianza = 0.000166
    # 5 vecinos -> Media = 0.9650 | Varianza = 0.000113
    # 10 vecinos -> Media = 0.9566 | Varianza = 0.000162
    # 100 vecinos -> Media = 0.8993 | Varianza = 0.000283
    # 1000 vecinos -> Media = 0.5275 | Varianza = 0.005013
    #print(f"El ratio_aciertos es: {ratio_aciertos}")

    # --- 5. Support Vector Machine (SVM)
    clasificador_SVC = SVC(kernel='linear', C=1.0)
    clasificador_SVC.fit(X_train, Y_train)


#calculamos la media y varianza con ratios obtenidos MLPC
media_ratio_aciertos_MLPC,varianza_ratio_aciertos_MLPC = calcular_media_varianza(vector_aciertos_MLPC)
print("#################### MLPC #########################################")
print(f"Ratios obtenidos= {vector_aciertos_MLPC}")
print(f"Media del ratio de aciertos= {media_ratio_aciertos_MLPC} |||| Varianza del ratio de aciertos= {varianza_ratio_aciertos_MLPC}")

print("########################## NBayes #################################")
media_ratio_aciertos_NBayes,varianza_ratio_aciertos_NBayes = calcular_media_varianza(vector_aciertos_NBayes)
print(f"Ratios obtenidos= {vector_aciertos_NBayes}")
print(f"Media del ratio de aciertos= {media_ratio_aciertos_NBayes} |||| Varianza del ratio de aciertos= {varianza_ratio_aciertos_NBayes}")

print("########################## LDA #################################")
media_ratio_aciertos_LDA,varianza_ratio_aciertos_LDA = calcular_media_varianza(vector_aciertos_LDA)
print(f"Ratios obtenidos= {vector_aciertos_LDA}")
print(f"Media del ratio de aciertos_LDA= {media_ratio_aciertos_LDA} |||| Varianza del ratio de aciertos_LDA= {varianza_ratio_aciertos_LDA}")

print("########################## KNN #################################")
media_ratio_aciertos_KNN,varianza_ratio_aciertos_KNN = calcular_media_varianza(vector_aciertos_KNN)
print(f"Ratios obtenidos= {vector_aciertos_KNN}")
print(f"Media del ratio de aciertos= {media_ratio_aciertos_KNN} |||| Varianza del ratio de aciertos= {varianza_ratio_aciertos_KNN}")