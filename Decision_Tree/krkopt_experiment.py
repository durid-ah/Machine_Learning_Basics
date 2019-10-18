import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

columns = ['w_k_file', 'w_k_rank', 'w_r_file', 'w_r_rank', 'b_k_file', 'b_k_rank', 'class']
krk = pd.read_csv("krkopt.data", header=None, names=columns)
