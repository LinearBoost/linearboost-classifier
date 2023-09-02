import numpy as np, time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


class SEFR(BaseEstimator):
    """
    This is the multiclass classifier version of the SEFR algorithm for Python
    based on https://github.com/sefr-classifier/sefr/blob/master/SEFR.py
    
    Also see: https://arxiv.org/abs/2006.04620
    """
    
    def __init__(self):
        """
        Initialize model class.
        """
        
        self.labels = np.array([])
        self.weights = np.array([])
        self.bias = np.array([])
        self.training_time = 0
        self.classes_ = np.array([])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoder = LabelEncoder()

        #self.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    def _sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(x))

    def fit(self, data_train, target_train, sample_weight=None):
        """
        Train the model.
        """
        
        self.labels = np.unique(target_train) # get all labels
        self.classes_ = self.labels
        self.weights = []
        self.bias = []
        self.training_time = 0
    
        if sample_weight is not None:
            sample_weight = np.array(sample_weight, dtype="float32")
        else:
            sample_weight = np.ones(len(target_train), dtype="float32")
            
        data_train = self.scaler.fit_transform(data_train)
        
        self.label_encoder.fit(target_train)
        encoded_labels = self.label_encoder.transform(target_train)

        start_time = time.monotonic_ns()
        data_train = np.array(data_train, dtype='float32')
        target_train = np.array(encoded_labels, dtype='int32')
        
        
        for label in self.labels: # train binary classifiers on each labels
            
            pos_labels = (target_train != label) # use "not the label" as positive class
            neg_labels = np.invert(pos_labels) # use the label as negative class
            
            pos_indices = data_train[pos_labels]
            neg_indices = data_train[neg_labels]
            
            avg_pos = np.average(pos_indices, axis=0, weights=sample_weight[pos_labels])
            avg_neg = np.average(neg_indices, axis=0, weights=sample_weight[neg_labels])
            
            weight = np.nan_to_num((avg_pos - avg_neg) / (avg_pos + avg_neg)) # calculate model weight of "not the label"
            #print(weight)
            
            weighted_scores = np.dot(data_train, weight)
            
            pos_score_avg = np.average(weighted_scores[pos_labels], weights=sample_weight[pos_labels])
            neg_score_avg = np.average(weighted_scores[neg_labels], weights=sample_weight[neg_labels])
            
            bias = -(neg_indices.size * pos_score_avg + # calculate weighted average of bias
                     pos_indices.size * neg_score_avg) / (neg_indices.size + pos_indices.size)
            
            self.weights.append(weight) # label weight
            self.bias.append(bias) # label bias
        
        self.weights = np.array(self.weights, dtype='float32')
        self.bias = np.array(self.bias, dtype='float32')
        self.training_time = time.monotonic_ns() - start_time


    def predict(self, new_data):
        """
        Predict labels of the new data.
        """
        new_data = self.scaler.transform(new_data)
        new_data = np.array(new_data, dtype='float32')

        # calculate weighted score + bias on each labels
        weighted_score = np.add(np.dot(self.weights, new_data.T).T, self.bias)
        #print(weighted_score)
        preds = self.labels[np.argmin(weighted_score, axis=1)]
        original_preds = self.label_encoder.inverse_transform(preds)
        return original_preds

    def predict_proba(self, new_data):
        """
        Predict class probabilities of the new data.
        """
        new_data = self.scaler.transform(new_data)
        new_data = np.array(new_data, dtype='float32')
        #print(new_data)

        # calculate weighted score + bias on each label
        weighted_score = np.add(np.dot(self.weights, new_data.T).T, self.bias)
        
        weighted_score = self._sigmoid(weighted_score)
        
        exp_scores = np.exp(weighted_score)
        
        class_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #class_probs = weighted_score / np.sum(weighted_score, axis=1, keepdims=True)
        #print(class_probs)
        
        return class_probs
    
    
    def get_params(self, deep=True): # for cross-validation
        return {}

def linboostclassifier(n_estimators=10):
    return GradientBoostingClassifier(init=SEFR(), n_estimators=n_estimators)
# ================================================================================

if __name__ == '__main__':

    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report

    # load Iris dataset
    # source: https://archive.ics.uci.edu/ml/datasets/iris
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                     header=None, names=('sepal length', 'sepal width', 'petal length', 'petal width', 'class'))
    
    # extract data and target
    X = df.drop(['class'], axis=1).to_numpy()
    y = df['class'].to_numpy()
    
    # encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = le.classes_

    # prepare training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # train model and predict labels
    clf = SEFR()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    cv_predicted = cross_val_predict(clf, X_train, y_train, cv=5)

    # view prediction results
    print('Training time:', clf.training_time, 'ns')
    print('Training CV score:', accuracy_score(y_train, cv_predicted).round(3))
    print('Test accuracy:', accuracy_score(y_test, predicted).round(3))
    print('')
    print('Test classification report:')
    print(classification_report(y_test, predicted, target_names=class_names))