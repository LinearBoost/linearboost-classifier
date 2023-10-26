import pandas as pd
import numpy as np
import matplotlib as plt
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import SefrBinary as sefr
from SefrBinary import linboostclassifier, LinBoostClassifier

import timeit

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
#from sklearn.ensemble import 

#df = pd.read_csv("D:/SEFR/Binary Classification/Datasets/GLI_85.csv", header = None)
#df = pd.read_csv("D:/SEFR/Binary Classification/Datasets/SMK_CAN_187.csv", header = None)
#df = pd.read_csv("D:/SEFR/Binary Classification/Datasets/Basehock.csv", header = None)
#df = pd.read_csv("D:/SEFR/Binary Classification/Datasets/Gisette.csv", header = None)
#df = pd.read_csv("D:/SEFR/Binary Classification/Datasets/Sonar.csv", header = None)
#df = pd.read_csv("./Datasets/GLI.csv", header = None)
#df = pd.read_csv("./Datasets/Banknote.csv", header = None)
#df = pd.read_csv("D:/Datasets/Big Data/Higgs/HIGGS - Normalized.csv", header = None)
#df = pd.read_csv("C:/Users/HP/Documents/SonarZ.csv", header = None)
#df = pd.read_csv("D:/Datasets/Big Data/SUSY/SUSY - Normalized.csv", header = None)
#df = pd.read_csv("D:/SEFR/Datasets/HeartAttackAnalysisPrediction.csv")
#df = pd.read_csv("D:/SEFR/Datasets/wdbc/wdbc.data")
#df = pd.read_csv("D:/SEFR/Datasets/Mushroom/agaricus-lepiota.data", header=None)

#df = pd.read_csv("D:/SEFR/Datasets/adult.data", header=None)

#df = pd.read_csv("D:/SEFR/Datasets/creditcard.csv")
df = pd.read_csv("D:/SEFR/Datasets/spambase.data", header=None)


print(df)
print(df.dtypes)

print(df.columns)

#exit()
#print(df[14])
#print(df.dtypes)
#exit()



# =============================================================================
# cat_features = [1, 3, 5, 6, 7, 8, 9, 13]#range(1, 23)
# cat_features = ['cp', 'restecg']
# for f in cat_features:
#     df_onehot = pd.get_dummies(df[f], prefix=f)
#     df = df.drop(f, axis=1)
#     df = pd.concat([df_onehot, df], axis=1)
# 
# #print(df)
# =============================================================================
#exit()

#df.columns = df.columns.astype(str)

#df = df.fillna(df.mean())





start = timeit.default_timer()

def classification_model(model, data, predictors, outcome):
    kf = KFold(n_splits=10, shuffle = True, random_state=100)
    fold= 0
    error = []
    errs = []
    for train_index, test_index in kf.split(data):

        X_train = data.iloc[train_index,:-1]
        #print(X_train)
        #print('salam')
        #exit()
        y_train = data.iloc[train_index, -1]
        X_test = data.iloc[test_index,:-1]
        y_test = data.iloc[test_index,-1]
# =============================================================================
#         
#         X_train = data.iloc[train_index,:-1]
#         y_train = data.iloc[train_index, -1]
#         X_test = data.iloc[test_index,:-1]
#         y_test = data.iloc[test_index,-1]
#         
# =============================================================================
        #print(X_train)
        #print(y_train)
        #adaboost_sefr = model
        #adaboost_sefr = AdaBoostClassifier(estimator=model, n_estimators=200, random_state=0, algorithm="SAMME")
        adaboost_sefr = model
        #bagging adaboost_sefr = BaggingClassifier(base_estimator=model, n_estimators=100, random_state=0)
        #adaboost_sefr = GradientBoostingClassifier(init=model, n_estimators=10)

        adaboost_sefr.fit(X_train, y_train)
        #model.fit(X_train, y_train)
        
        y_pred = adaboost_sefr.predict(X_test)
        
        
        acc = metrics.accuracy_score(y_pred, y_test)
        err = metrics.f1_score(y_pred, y_test, average = 'macro')
        
        errs.append(err)
        error.append(acc)
        fold = fold + 1
        stop = timeit.default_timer()
        #print('Time: ', stop - start)

        
    print("Cross-Validation Accuracy Score: %s" % "{0:.2%}".format(np.mean(error)))
    print("Cross-Validation F1 Score: %s" % "{0:.2%}".format(np.mean(errs)))

predictor_var = []
outcome_var = 0

for i in [5, 10, 20, 50, 100, 200, 500, 1000]:
    #model = XGBClassifier(n_estimators=i)
    #model = lgb.LGBMClassifier(n_estimators=i, verbose=-1)
    #model = CatBoostClassifier(n_estimators=i, verbose=0)
    #model = tree.DecisionTreeClassifier()
    #model = RandomForestClassifier()
    #model = GaussianNB()
    #model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    #model = svm.SVC(kernel='linear')
    #model = LogisticRegression(max_iter=500)
    #model = sefr.SEFR()
    #model = linboostclassifier()
    #model = KNeighborsClassifier()
    model = LinBoostClassifier(n_estimators=i, algorithm="SAMME.R")
    print(i)
    classification_model(model, df, predictor_var, outcome_var)
    stop = timeit.default_timer()
    
    print("Time: {:.2f}".format(stop - start))
    print('*'*65)
    
    
    
