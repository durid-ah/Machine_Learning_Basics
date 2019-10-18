from .KnnClassifier import KnnClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

x = np.array([3, 6])
y = np.array([5, 2])
data = np.array([[2, 3], [3, 4], [5, 7], [2, 7], [3, 2], [1, 2], [9, 3], [4, 1]])
animals = ["dog", "cat", "cat", "bird", "fish", "fish", "dog", "cat", "dog"]

number_to_target = {
                    'dog': 0,
                    'cat': 1,
                    'bird': 2,
                    'fish': 3
}

target = []

for item in animals:
    target.append(number_to_target[item])

print(target)

var = KnnClassifier()
num = var.classify(2, target, [[2, 3]], data)
print(num)

######################################################

iris = datasets.load_iris()
train_data_set, test_data_set, train_target_set, test_target_set = model_selection.train_test_split(iris.data,
                                                                                                    iris.target,
                                                                                                    shuffle=True,
                                                                                                    train_size=0.3)

iris_result = var.classify(5, train_target_set, test_data_set, train_data_set)
print("Accuracy:", metrics.accuracy_score(test_target_set, iris_result))
print(metrics.confusion_matrix(test_target_set, iris_result))
print(metrics.classification_report(test_target_set, iris_result))

accuracy_set = []

for i in range(1, 50):
    result_set = var.classify(i, train_target_set, test_data_set, train_data_set)
    accuracy_set.append((metrics.accuracy_score(test_target_set, result_set)) * 100)

[print("%.2f" % item) for item in accuracy_set]

######################################################

scaler = StandardScaler()
scaler.fit(train_data_set)

train_data_set = scaler.transform(train_data_set)
test_data_set = scaler.transform(test_data_set)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_data_set, train_target_set)

y_pred = classifier.predict(test_data_set)
print("Accuracy:", metrics.accuracy_score(test_target_set, y_pred))

print(metrics.confusion_matrix(test_target_set, y_pred))
print(metrics.classification_report(test_target_set, y_pred))

#########################################################

breast_cancer = datasets.load_breast_cancer()

train_data_set, test_data_set, train_target_set, test_target_set = model_selection.train_test_split(breast_cancer.data,
                                                                                                    breast_cancer.target,
                                                                                                    shuffle=True,
                                                                                                    train_size=0.3)

breast_cancer_result = var.classify(3, train_target_set, test_data_set, train_data_set)
print("Accuracy:", metrics.accuracy_score(test_target_set, breast_cancer_result))

##########################################################
