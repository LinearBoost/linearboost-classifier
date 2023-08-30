import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

class SEFR(BaseEstimator):
    def __init__(self):
        self.weights = []
        self.bias = 0
        self.classes_ = np.array([0, 1])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoder = LabelEncoder()

    def fit(self, train_predictors, train_target, sample_weight=None):
        """
        This is used for training the classifier on data.
        Parameters
        ----------
        train_predictors : float, either list or numpy array
            are the main data in DataFrame
        train_target : integer, numpy array
            labels, should consist of 0s and 1s
        """
        train_predictors = self.scaler.fit_transform(train_predictors)
        
        self.label_encoder.fit(train_target)
        encoded_labels = self.label_encoder.transform(train_target)
        #print(train_predictors)
        
        X = np.array(train_predictors, dtype="float32")
        y = np.array(encoded_labels, dtype="int32")

        if sample_weight is not None:
            sample_weight = np.array(sample_weight, dtype="float32")
        else:
            sample_weight = np.ones(len(y), dtype="float32")

        pos_labels = np.sign(y) == 1
        neg_labels = np.invert(pos_labels)

        pos_indices = X[pos_labels, :]
        neg_indices = X[neg_labels, :]
        
        
        
        avg_pos = np.average(pos_indices, axis=0, weights=sample_weight[pos_labels])
        avg_neg = np.average(neg_indices, axis=0, weights=sample_weight[neg_labels])

        self.weights = (avg_pos - avg_neg) / (avg_pos + avg_neg + 0.0000001)

        sum_scores = np.dot(X, self.weights)

        pos_label_count = np.count_nonzero(y)
        neg_label_count = y.shape[0] - pos_label_count

        pos_score_avg = np.average(sum_scores[y == 1], weights=sample_weight[y == 1])
        neg_score_avg = np.average(sum_scores[y == 0], weights=sample_weight[y == 0])

        self.bias = (neg_label_count * pos_score_avg + pos_label_count * neg_score_avg) / (neg_label_count + pos_label_count)
        #print(sample_weight)
        #print(self.weights, self.bias)
        
        
        
    def predict(self, test_predictors):
        """
        This is for prediction. When the model is trained, it can be applied on the test data.
        Parameters
        ----------
        test_predictors: either list or ndarray, two dimensional
            the data without labels in
        Returns
        ----------
        predictions in numpy array
        """
        X = self.scaler.transform(test_predictors)
        if isinstance(test_predictors, list):
            X = np.array(test_predictors, dtype="float32")

        temp = np.dot(X, self.weights)
        preds = np.where(temp <= self.bias, 0 , 1)
        original_preds = self.label_encoder.inverse_transform(preds)
        return original_preds 
    
    def predict_proba(self, test_predictors):
        """
        This is for prediction probabilities.
        Parameters
        ----------
        test_predictors: either list or ndarray, two dimensional
            the data without labels in
        Returns
        ----------
        prediction probabilities in numpy array
        """
        X = self.scaler.transform(test_predictors)
        if isinstance(test_predictors, list):
            X = np.array(test_predictors, dtype="float32")
            
        linear_output = np.dot(X, self.weights) - self.bias
        
        #print('llllll', linear_output)
        #exit()
        #pred_proba = 1 / (1 + np.exp(-linear_output))
        #pred_proba = np.exp(linear_output) / (np.exp(linear_output) + np.exp(-linear_output))
        #print(pred_proba)
        #return np.column_stack((1 - pred_proba, pred_proba))
    

        temp = np.dot(X, self.weights)
        score = (temp - self.bias) / np.linalg.norm(self.weights)
        pred_proba = 1 / (1 + np.exp(-score))
        return np.column_stack((1 - pred_proba, pred_proba))


def linboostclassifier(n_estimators=200, random_state=0, algorithm="SAMME"):
    return AdaBoostClassifier(estimator=SEFR(), n_estimators=n_estimators, random_state=random_state, algorithm=algorithm)