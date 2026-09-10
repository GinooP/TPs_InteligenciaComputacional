import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


def calcular_media_varianza(vector):
    N=len(vector)
    media= np.sum(vector)/N
    varianza= np.sum((vector - media)**2)/N
    return media,varianza

print("####################### Utilizando KFold con 5 particiones ###################################")
wine = load_wine() #cargamos los datos del conjunto digits

X= wine.data
Y= wine.target

print(len(X))
print(len(Y))

eta =0.001
epocas=1000
tol=1e-4

fk= KFold(n_splits=5,shuffle=True)#lo dejamos con shufle en false

vector_aciertos_MLPC=np.array([])
vector_aciertos_NBayes=np.array([])
vector_aciertos_Tree_Bagging=np.array([])
vector_aciertos_KNN=np.array([])
vector_aciertos_LDA=np.array([])

vector_aciertos_ADA=np.array([])
for i,(index_train,index_test) in enumerate(fk.split(X)):
    print(f"Fold= {i}")

    X_train = X[index_train]
    Y_train = Y[index_train]

    X_test = X[index_test]
    Y_test = Y[index_test]

    ###### Perceptron multicapa #################################
    #definimos el modelo dentro del for para entrenar el mismo modelo con cada fold
    # modelo = MLPClassifier(hidden_layer_sizes=(100,100 ),activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
    #                    random_state=42, tol=tol, verbose=True,n_iter_no_change=epocas,learning_rate='adaptive',) #para que no muestre en consola todo el proceso

    # #nota: una optimizacion posible es hacer modelo.fit(X[index_train],Y[index_train]) y lo mismo en test para no ocupar tanta memoria
    # modelo.fit(X_train,Y_train)

    # ratio_aciertos= modelo.score(X_test,Y_test)
    # vector_aciertos_MLPC =np.append(vector_aciertos_MLPC,ratio_aciertos)

    clf = BaggingClassifier(estimator=SVC(),n_estimators=10, random_state=0,)
    clf.fit(X_train,Y_train)
    
    vector_aciertos_Tree_Bagging=np.append(vector_aciertos_Tree_Bagging,clf.score(X_test,Y_test))

    clf_AB = AdaBoostClassifier(estimator=SVC(kernel='linear',C=1.0), n_estimators=10, learning_rate=1.0, random_state=None)
    clf_AB.fit(X_train,Y_train)
    vector_aciertos_ADA=np.append(vector_aciertos_ADA,clf_AB.score(X_test,Y_test))
    

# #calculamos la media y varianza con ratios obtenidos MLPC
# media_ratio_aciertos_MLPC,varianza_ratio_aciertos_MLPC = calcular_media_varianza(vector_aciertos_MLPC)
# print("#################### MLPC #########################################")
# print(f"Ratios obtenidos= {vector_aciertos_MLPC}")
# print(f"Media del ratio de aciertos= {media_ratio_aciertos_MLPC} |||| Varianza del ratio de aciertos= {varianza_ratio_aciertos_MLPC}")

media_ratio_aciertos_Tree_Bagging,varianza_ratio_aciertos_Tree_Bagging = calcular_media_varianza(vector_aciertos_Tree_Bagging)
print("#################### bagging #########################################")
print(f"Ratios obtenidos= {vector_aciertos_Tree_Bagging}")
print(f"Media del ratio de aciertos= {media_ratio_aciertos_Tree_Bagging} |||| Varianza del ratio de aciertos= {varianza_ratio_aciertos_Tree_Bagging}")

media_ratio_aciertos_ADA,varianza_ratio_aciertos_ADA = calcular_media_varianza(vector_aciertos_ADA)
print("#################### bagging #########################################")
print(f"Ratios obtenidos= {vector_aciertos_ADA}")
print(f"Media del ratio de aciertos= {media_ratio_aciertos_ADA} |||| Varianza del ratio de aciertos= {varianza_ratio_aciertos_ADA}")