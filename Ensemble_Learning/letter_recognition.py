import pandas as pd
from sklearn import model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC


categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
categories = list(map(lambda x: x.upper(), categories))
number = [i for i in range(26)]
alpha_to_int = {}
for i in number:
    alpha_to_int[categories[i]] = i

letters = pd.read_csv("../Datasets/letter-recognition.data", header=None, na_values='?')
letters.replace({
    0: alpha_to_int
}, inplace=True)

dataset = letters[range(1,16)]
labels = letters[0]

train_data_set, test_data_set, train_target_set, test_target_set = model_selection.train_test_split(dataset,
                                                                                                    labels,
                                                                                                    shuffle=True,
                                                                                                    train_size=0.3)


# scaling the data
scaler = StandardScaler()
scaler.fit(train_data_set)

train_data_set = scaler.transform(train_data_set)
test_data_set = scaler.transform(test_data_set)


# KNN classification
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_data_set, train_target_set)

y_pred = classifier.predict(test_data_set)
print("KNN Accuracy:", metrics.accuracy_score(test_target_set, y_pred))

# Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(train_data_set, train_target_set)
predicted = clf.predict(test_data_set)
print("Decision Tree Accuracy:", metrics.accuracy_score(test_target_set, predicted))

# GaussianNB
classifier = GaussianNB()
classifier.fit(train_data_set, train_target_set)
prediction_result = classifier.predict(test_data_set)

print("GaussianNB Accuracy:", metrics.accuracy_score(test_target_set, prediction_result))

# AdaBoost
dtc = DecisionTreeClassifier()
svc = SVC(probability=True, kernel='linear')
abc = AdaBoostClassifier(n_estimators=50,
                         base_estimator=dtc,
                         learning_rate=1)

model = abc.fit(train_data_set, train_target_set)
y_pred = model.predict(test_data_set)
print("AdaBoost Accuracy:",metrics.accuracy_score(test_target_set, y_pred))

# Bagging
dtc2 = DecisionTreeClassifier()
bag_model = BaggingClassifier(base_estimator=dtc2,
                              n_estimators=100,
                              bootstrap=True)

bag_model = bag_model.fit(train_data_set, train_target_set)
bagging_pred = bag_model.predict(test_data_set)
print("Bagging Accuracy:",
      metrics.accuracy_score(test_target_set, bagging_pred))

