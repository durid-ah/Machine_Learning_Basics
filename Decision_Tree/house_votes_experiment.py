import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

columns = \
    ['label', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
     'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
     'aid-to-nicaraguan-contras', ' mx-missile', 'immigration', 'synfuels-corporation-cutback',
     'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
     'export-administration-act-south-africa']

votes = pd.read_csv("house-votes-84.data", header=None, names=columns, na_values='?')
votes = votes.dropna()

features = \
    ['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
     'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
     'aid-to-nicaraguan-contras', ' mx-missile', 'immigration', 'synfuels-corporation-cutback',
     'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports',
     'export-administration-act-south-africa']

value_encoder = preprocessing.LabelEncoder()
value_encoder.fit(['n', 'y'])

for column in features:
    votes[column] = value_encoder.transform(votes[column])

dataset = votes[features]
labels = votes.label

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
                class_names=['0', '1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_svg('votes.svg')
Image(graph.create_svg())