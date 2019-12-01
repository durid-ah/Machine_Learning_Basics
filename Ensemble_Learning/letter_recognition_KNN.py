import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

