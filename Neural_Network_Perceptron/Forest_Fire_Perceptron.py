import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv")

months =\
    {"jan": 0, "feb": 1, "mar": 2, "apr": 3,"may": 4, "jun": 5, "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11}

day = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

df.replace({
    "month": months,
    "day": day
}, inplace=True)

Features = ["X", "Y", "month", "day", ]

dataset = df[Features]
labels = df.area

print(dataset)

train_data_set, test_data_set, train_target_set, test_target_set = train_test_split(dataset,
                                                                                    labels,
                                                                                    shuffle=True,
                                                                                    train_size=0.3)

scaler = StandardScaler()
scaler.fit(train_data_set)

train_data_set = scaler.transform(train_data_set)
test_data_set = scaler.transform(test_data_set)

mlp = MLPRegressor(max_iter=1000)
mlp.fit(train_data_set, train_target_set)
predictions = mlp.predict(test_data_set)
