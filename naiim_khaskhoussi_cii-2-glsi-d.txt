
"""
Soit la base de données ‘Dataset_spine.csv’ contient des informations sur la colonne vertébrale.
Elle est composée de deux classe classe ‘normal ‘ et classe ‘Abnormal’
1) Donner le nombre des observations dans chaque classe.
2) Effectuer le prétraitement nécessaire pour cette base de données.
3) Diviser la base de données nettoyée en deux parties : 2/3 pour l’apprentissage et 1/3 pour le test
4) Choisir trois modèles pour la discrimination (classification) entre ces deux classes (Normal et abnormal). Et justifier votre réponse.
5) Construire les trois modèles en utilisant le 2/3 de la base.
6) Tester ces trois modèles choisis sur le 1/3 de la base.
7) Calculer la performance de trois modèles choisis en utilisant tous les protocoles expérimentaux de la classification.
8) Interpréter les résultats obtenus.
"""

import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# precision_score, recall_score

data_source = 'Dataset_spine.csv'
dataset = pd.read_csv(data_source)

# (1) Donner le nombre des observations dans chaque classe.
print("\nAnalyse des observations")
print("Total: {} observations".format(dataset.shape[0]))
print(dataset['Class_att'].value_counts())

# (2) Effectuer le prétraitement nécessaire pour cette base de données
# extraction du vecteur Labels 
y = dataset ['Class_att']
# supprimer le vecteur label du dataset
dataset.drop(['Class_att'], axis=1, inplace=True)
print("\nPre-processing done.\n")

# (3) Diviser la base de données nettoyée en deux parties : 2/3 pour l’apprentissage et 1/3 pour le test
x_train, x_test ,y_train ,y_test = train_test_split(dataset, y, test_size=0.33, random_state=0)
print("Train-Test split done.\n")

# (4) Choisir trois modèles pour la discrimination (classification) entre ces deux classes (Normal et abnormal). Et justifier votre réponse.
# Justification: Apprentissage suppérvisé, classification
models = [
    {"Model": svm.SVC(kernel='linear')},
    {"Model": LogisticRegression(solver='lbfgs', max_iter=200)},
    {"Model": DecisionTreeClassifier()}
]
print("Models creation done.\n")

# (5) Construire les trois modèles en utilisant le 2/3 de la base.
for model_details in models:

    # create and train the selected model
    model = model_details["Model"]

    print("Working with: {}".format(model))
    model.fit(x_train, y_train)

    # (6) Tester ces trois modèles choisis sur le 1/3 de la base.
    y_pred = model.predict(x_test)

    # (7) Calculer la performance de trois modèles choisis en utilisant tous les protocoles expérimentaux de la classification.
    model_details["Score"] = model.score(dataset, y)
    model_details["Accuracy"] = accuracy_score(y_test, y_pred)
    model_details["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
    #model_details["Precision Score"] = precision_score(y_test, y_pred)
    #model_details["Recall Score"] = recall_score(y_test, y_pred)
    #print(classification_report(y_test, y_pred))
    print("Done.\n")

# (8) Interpréter les résultats obtenus
results_df = pd.DataFrame(models)
print(results_df)

"""
Interprétation:
Le modèle DecisionTreeClassifier donne un score mieux important mais avec faible accuracy
Le modèle SVC donne les résulats le plus précise pour notre dataset
En génerale on n'est pas ni dans l'overfitting ni l'underfitting, et on avaient des bon résulats de prédictions
"""
