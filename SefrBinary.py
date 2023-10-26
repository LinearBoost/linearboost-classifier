import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

class SEFR(BaseEstimator):
    def __init__(self):
        self.weights = []
        self.bias = 0
        self.classes_ = np.array([0, 1])
        #self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoder = LabelEncoder()
        self.max = 0
        self.min = 0

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
        #train_predictors = self.scaler.fit_transform(train_predictors)
        
        #self.label_encoder.fit(train_target)
        #encoded_labels = self.label_encoder.transform(train_target)
        #print(train_predictors)
        
        X = np.array(train_predictors, dtype="float32")
        #y = np.array(encoded_labels, dtype="int32")
        y = np.array(train_target, dtype="int32")

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
        
        self.max = max(sum_scores)
        self.min = min(sum_scores)
        
        

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
        #X = self.scaler.transform(test_predictors)
        X = test_predictors
        if isinstance(test_predictors, list):
            X = np.array(test_predictors, dtype="float32")

        temp = np.dot(X, self.weights)
        preds = np.where(temp <= self.bias, 0 , 1)
        #original_preds = self.label_encoder.inverse_transform(preds)
        #return original_preds 
        return preds
    
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
        X = test_predictors
        if isinstance(test_predictors, list):
            X = np.array(test_predictors, dtype="float32")
            
        linear_output = np.dot(X, self.weights) - self.bias
        
        #print('llllll', linear_output)
        #exit()
# =============================================================================
#         pred_proba = 1 / (1 + np.exp(-linear_output))
#         pred_proba = np.exp(linear_output) / (np.exp(linear_output) + np.exp(-linear_output))
#         #print(pred_proba)
#         return np.column_stack((1 - pred_proba, pred_proba))
# =============================================================================
    

        temp = np.dot(X, self.weights)
        score = (temp - self.bias) / np.linalg.norm(self.weights)
        #score = (temp - self.bias) / self.max
        pred_proba = 1 / (1 + np.exp(-score))
        #pred_proba = (np.tanh(score) + 1) / 2
        #print(pred_proba)
        return np.column_stack((1 - pred_proba, pred_proba))


        
    
        # to be tested later
        #score = ((temp - self.bias) - self.min) / (self.max - self.min)
        #pred_proba = score
        #print(pred_proba)
        #return np.column_stack((1 - pred_proba, pred_proba))


class LinBoostClassifier(AdaBoostClassifier):
    def __init__(self, n_estimators=200, learning_rate=1.0, algorithm='SAMME', random_state=9):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoder = LabelEncoder()
        super().__init__(estimator=SEFR(), n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=random_state)


    def fit(self, X, y, sample_weight=None):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.label_encoder.fit(y)
        y = self.label_encoder.transform(y)
        return super().fit(X, y, sample_weight)
    
    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = super().predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        return super().predict_proba(X)
    
    
def linboostclassifier(n_estimators=200, random_state=0, algorithm="SAMME"):
    return AdaBoostClassifier(estimator=SEFR(), n_estimators=n_estimators, random_state=random_state, algorithm=algorithm)