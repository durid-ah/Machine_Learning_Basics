import pandas as pd
from sklearn import model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC

soybeans = pd.read_csv("../Datasets/soybean-large.data", header=None, na_values='?')
soybeans = soybeans.dropna()

dataset = soybeans[range(1,36)]
labels = soybeans[0]

train_data_set, test_data_set, train_target_set, test_target_set = model_selection.train_test_split(dataset,
                                                                                                    labels,
                                                                                                    shuffle=True,
                                                                                                    train_size=0.3)

# scaling the data
scaler = StandardScaler()
scaler.fit(train_data_set)

train_data_set = scaler.transform(train_data_set)
test_data_set = scaler.transform(test_data_set)

# GaussianNB
gnb = GaussianNB()
gnb.fit(train_data_set, train_target_set)
gnb_pred = gnb.predict(test_data_set)

print("GaussianNB Accuracy:", metrics.accuracy_score(test_target_set, gnb_pred))

# SVC classifier
svc = SVC(probability=True, kernel='linear')
svc.fit(train_data_set, train_target_set)
svc_pred = svc.predict(test_data_set)
print("SVC Accuracy:",
      metrics.accuracy_score(test_target_set, svc_pred))

# Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(train_data_set, train_target_set)
predicted = clf.predict(test_data_set)
print("Decision Tree Accuracy:", metrics.accuracy_score(test_target_set, predicted))

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
