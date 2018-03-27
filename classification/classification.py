'''
#########################################################################
#   File Name       : Classification.py
#########################################################################
#   Project/Product : All
#   Title           : IRIS Classification
#   Author          : Guoan.Li
#########################################################################
#   Description     : list 7 kinds of classification algorithm
#
#########################################################################
#   Revision History:
# 
#   Version     Date          Initials      CR#          Descriptions
#   ---------   ----------    ------------  ----------   ---------------
#   1.0         27/03/2018    Guoan.Li      N/A          Original
#########################################################################
'''

from sklearn import svm, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

train_x, test_x, train_y, test_y = train_test_split(iris.data, 
                                                    iris.target, 
                                                    test_size = 0.4,
                                                    random_state=0)

C0 = 1.0  # SVM globle parameter

models = (svm.SVC(kernel = 'linear', C = C0),
          svm.SVC(kernel = 'rbf', gamma=0.6, C = C0),
          svm.SVC(kernel = 'poly', degree=3, C = C0),
          svm.LinearSVC(),
          DecisionTreeClassifier(max_depth=3),
          LogisticRegression(C = C0),
          RandomForestClassifier(max_depth = 3, random_state = 0))

for i in range(len(models)):
    models[i].fit(train_x, train_y)
    print(i,models[i].__class__.__name__, 'train score: ',models[i].score(train_x, train_y))
    print(i,models[i].__class__.__name__, 'test score: ',models[i].score(test_x, test_y))