'''
用机器学习模型对数据进行分类
'''

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from semg_dataloader import dataloader_ml


if __name__ == '__main__':

    X, Y = dataloader_ml(dataroot='processed_data/slice_data')
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print(len(X_train))
    classifier = DecisionTreeClassifier()
    # classifier = SVC()
    classifier = classifier.fit(X_train, y_train)

    preds = classifier.predict(X_test)

    preds = np.array(preds)
    y_test = np.array(y_test)
    acc = (preds==y_test).sum() / y_test.size

    print('acc: {:04f}'.format(acc))
