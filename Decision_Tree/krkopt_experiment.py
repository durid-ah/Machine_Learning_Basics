import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

classEncoding = {
    'draw': -1,
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16}

columns = ['w_k_file', 'w_k_rank', 'w_r_file', 'w_r_rank', 'b_k_file', 'b_k_rank', 'label']
krk = pd.read_csv("krkopt.data", header=None, names=columns)

krk.replace({
    "label": classEncoding
})

encoder = preprocessing.LabelEncoder()
encoder.fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
krk.w_k_file = encoder.transform(krk.w_k_file)
krk.w_r_file = encoder.transform(krk.w_r_file)
krk.b_k_file = encoder.transform(krk.b_k_file)

features = ['w_k_file', 'w_k_rank', 'w_r_file', 'w_r_rank', 'b_k_file', 'b_k_rank']
dataset = krk[features]
labels = krk.label

train_data_set, test_data_set, train_target_set, test_target_set = train_test_split(dataset,
                                                                                    labels,
                                                                                    shuffle=True,
                                                                                    train_size=0.3)

clf = DecisionTreeClassifier()
clf = clf.fit(train_data_set, train_target_set)
predicted = clf.predict(test_data_set)

print("Accuracy:", metrics.accuracy_score(test_target_set, predicted))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=features,
                class_names=['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_svg('krkopt.svg')
Image(graph.create_svg())