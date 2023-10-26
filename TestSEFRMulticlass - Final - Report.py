import pandas as pd
import numpy as np
import matplotlib as plt
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import tree

from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import SEFRMulticlass as sefr
from SEFRMulticlass import LinBoostClassifier



import timeit

#df = pd.read_csv("./Datasets/CNAE-9.csv", header = None)
#df = pd.read_csv("D:/SEFR/Multiclass Classification/Datasets/Wave5000.csv", header = None)
#df = pd.read_csv("D:/SEFR/Multiclass Classification/Datasets/Semeion.csv", header = None)
#df = pd.read_csv("./Datasets/MNIST.csv", header = None)
#print(df.iloc[:,1])


df = df.fillna(df.mean())

cat_features = ['cp', 'restecg']
for f in cat_features:
    df_onehot = pd.get_dummies(df[f], prefix=f)
    df = df.drop(f, axis=1)
    df = pd.concat([df_onehot, df], axis=1)

#print(df.columns)

df = df.fillna(df.mean())



start = timeit.default_timer()

def classification_model(model, data, predictors, outcome):
    kf = KFold(n_splits=10, shuffle = True, random_state=100)
    fold= 0
    error = []
    errs = []
    for train_index, test_index in kf.split(data):
# =============================================================================
#         X_train = data.iloc[train_index,1:]
#         y_train = data.iloc[train_index, 0]
#         X_test = data.iloc[test_index,1:]
#         y_test = data.iloc[test_index,0]
# =============================================================================
        
        X_train = data.iloc[train_index,:-1]
        y_train = data.iloc[train_index, -1]
        X_test = data.iloc[test_index,:-1]
        y_test = data.iloc[test_index,-1]
        
        #print(y_test)
        

        #adaboost_sefr = AdaBoostClassifier(estimator=model, n_estimators=2000, random_state=0, algorithm="SAMME")
        #bagging adaboost_sefr = BaggingClassifier(base_estimator=model, n_estimators=100, random_state=0)
        #adaboost_sefr = GradientBoostingClassifier(init=model, n_estimators=200)
        adaboost_sefr = model
        #adaboost_sefr = model

        adaboost_sefr.fit(X_train, y_train)
        #model.fit(X_train, y_train)
        y_pred = adaboost_sefr.predict(X_test)
        
        #y_pred = adaboost_sefr.predict(X_test)
        
        acc = metrics.accuracy_score(y_pred, data.iloc[test_index,0])
        err = metrics.f1_score(y_pred, data.iloc[test_index,0], average = 'macro')
        errs.append(err)
        error.append(acc)
        print(err)
        print(acc)
        
        fold = fold + 1
        stop = timeit.default_timer()
        #print('Time: ', stop - start)

        
    print("Cross-Validation Accuracy Score: %s" % "{0:.2%}".format(np.mean(error)))
    print("Cross-Validation F1 Score: %s" % "{0:.2%}".format(np.mean(errs)))

predictor_var = []
outcome_var = 0


for i in [5, 10, 20, 50, 100, 200, 500, 1000]    :
    #model = lgb.LGBMClassifier()
    model = XGBClassifier(n_estimators=i)
    #model = CatBoostClassifier(iterations=50, loss_function="MultiClass")
    #model = tree.DecisionTreeClassifier()
    #model = RandomForestClassifier()
    #model = GaussianNB()
    #model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    #model = sefr.SEFR()
    #model = linboostclassifier()
    #model = LinBoostClassifier(n_estimators=5)
    
    print(i)
    classification_model(model, df, predictor_var, outcome_var)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print('*'*65)



