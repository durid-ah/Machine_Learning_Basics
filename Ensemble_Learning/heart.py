import pandas as pd
from sklearn import model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC

columns =\
    ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
     "ca", "thal", "target"]

heart = pd.read_csv("../Datasets/heart.csv", na_values='?')

dataset = heart[columns[:-1]]
labels = heart[columns[-1]]

train_data_set, test_data_set, train_target_set, test_target_set = model_selection.train_test_split(dataset,
                                                                                                    labels,
                                                                                                    shuffle=True,
                                                                                                    train_size=0.3)

# scaling the data
scaler = StandardScaler()
scaler.fit(train_data_set)

train_data_set = scaler.transform(train_data_set)
test_data_set = scaler.transform(test_data_set)

svc = SVC(probability=True, kernel='linear')
svc.fit(train_data_set, train_target_set)
svc_pred = svc.predict(test_data_set)

print("SVC Accuracy:",
      metrics.accuracy_score(test_target_set, svc_pred))
