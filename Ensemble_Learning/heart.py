import pandas as pd
from sklearn import model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC

heart = pd.read_csv("../Datasets/heart.csv", na_values='?')
