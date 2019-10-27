import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


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


