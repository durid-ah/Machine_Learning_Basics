import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


columns = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree',
           'age', 'label']

pima = pd.read_csv("../Datasets/pima-indians-diabetes.csv", header=None, names=columns)

features = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']

dataset = pima[features]
labels = pima.label

train_data_set, test_data_set, train_target_set, test_target_set = train_test_split(dataset,
                                                                                    labels,
                                                                                    shuffle=True,
                                                                                    train_size=0.3)

scaler = StandardScaler()
scaler.fit(train_data_set)

train_data_set = scaler.transform(train_data_set)
test_data_set = scaler.transform(test_data_set)

mlp = MLPClassifier()
mlp.fit(train_data_set, train_target_set)
predictions = mlp.predict(test_data_set)

print(confusion_matrix(test_target_set,predictions))
print(classification_report(test_target_set,predictions))
print("Accuracy:", accuracy_score(test_target_set, predictions))

mlp_tuple = (100,)

print("Number of neurons:")

for k in range(1, 50):
    classifier = MLPClassifier(hidden_layer_sizes=(k,), max_iter=1000)
    classifier.fit(train_data_set, train_target_set)
    pred = classifier.predict(test_data_set)
    print("Accuracy:", accuracy_score(test_target_set, pred), k)

for i in range(1000, 2001, 500):
    for j in range(3):
        classifier = MLPClassifier(hidden_layer_sizes=mlp_tuple, max_iter=i)
        classifier.fit(train_data_set, train_target_set)
        pred = classifier.predict(test_data_set)
        print("Accuracy:", accuracy_score(test_target_set, pred), i, j)
        mlp_tuple = mlp_tuple + (100,)
