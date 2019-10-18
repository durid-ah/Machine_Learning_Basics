import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# TODO: The voters dataset is classified in the first column

columns = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree',
           'age', 'label']

pima = pd.read_csv("pima-indians-diabetes.csv", header=None, names=columns)

features = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']

dataset = pima[features]
labels = pima.label

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
                special_characters=True, feature_names=features, class_names=['0', '1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())