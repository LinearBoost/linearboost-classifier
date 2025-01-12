import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    QuantileTransformer,
    PowerTransformer,
    RobustScaler,
    Normalizer,
    StandardScaler,
    MaxAbsScaler
)
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from scipy.stats import boxcox

class SEFR(BaseEstimator):
    def __init__(self):
        self.weights = []
        self.bias = 0
        self.classes_ = np.array([0, 1])
        self.label_encoder = LabelEncoder()

    def fit(self, train_predictors, train_target, sample_weight=None):
        """
        This is used for training the classifier on data.
        """
        X = np.array(train_predictors, dtype="float32")
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

        self.bias = (neg_label_count * pos_score_avg + pos_label_count * neg_score_avg) / (
                    neg_label_count + pos_label_count)

    def predict(self, test_predictors):
        """
        This is for prediction. When the model is trained, it can be applied on the test data.
        """
        X = test_predictors
        if isinstance(test_predictors, list):
            X = np.array(test_predictors, dtype="float32")

        temp = np.dot(X, self.weights)
        preds = np.where(temp <= self.bias, 0, 1)
        return preds

    def predict_proba(self, test_predictors):
        """
        This is for prediction probabilities.
        """
        X = test_predictors
        if isinstance(test_predictors, list):
            X = np.array(test_predictors, dtype="float32")

        linear_output = np.dot(X, self.weights) - self.bias

        temp = np.dot(X, self.weights)
        score = (temp - self.bias) / np.linalg.norm(self.weights)
        pred_proba = 1 / (1 + np.exp(-score))
        return np.column_stack((1 - pred_proba, pred_proba))


class LinearBoostClassifier(AdaBoostClassifier):
    def __init__(self, n_estimators=200, learning_rate=1.0, algorithm='SAMME', scaler="minmax",
                 random_state=9, class_weight=None, loss_function=None):
        self.scaler = scaler
        self.scale_of_classes = 1.0  # Initialize to 1.0
        self.transformer = None
        self.minmaxscaler = MinMaxScaler(feature_range=(0, 1))
        self.label_encoder = LabelEncoder()
        self.loss_function = loss_function
        self.class_weight = class_weight # Added for class weights

        # Dictionary to store scalers for easy access
        self.scalers = {
            'quantile-uniform': QuantileTransformer(output_distribution='uniform', ignore_implicit_zeros=True),
            'quantile-normal': QuantileTransformer(output_distribution='normal', ignore_implicit_zeros=True),
            'normalizer-l1': Normalizer(norm='l1'),
            'normalizer-l2': Normalizer(norm='l2'),
            'normalizer-max': Normalizer(norm='max'),
            'standard': StandardScaler(),
            'power': PowerTransformer(method='yeo-johnson'),
            'maxabs': MaxAbsScaler(),
            'robust': RobustScaler()
        }

        super().__init__(estimator=SEFR(), n_estimators=n_estimators, learning_rate=learning_rate,
                         algorithm=algorithm, random_state=random_state)

    def fit(self, X, y, sample_weight=None):
        # Apply Scaler
        if self.scaler in self.scalers:
            self.transformer = self.scalers[self.scaler].fit(X)
            X_mapped = self.transformer.transform(X)
            self.minmaxscaler.fit(X_mapped)
            X_transformed = self.minmaxscaler.transform(X_mapped)
        else:
            self.minmaxscaler.fit(X)
            X_transformed = self.minmaxscaler.transform(X)

        # Label encoding for the target variable
        self.label_encoder.fit(y)
        y = self.label_encoder.transform(y)

        # Handle class weights
        if self.class_weight == "balanced":
            n_classes = len(np.unique(y))
            n_samples = len(y)
            class_counts = np.bincount(y)
            class_weights_arr = n_samples / (n_classes * class_counts)
            
            if sample_weight is None:
                sample_weight = np.array([class_weights_arr[label] for label in y])
            else:
                sample_weight = sample_weight * np.array([class_weights_arr[label] for label in y])
        elif isinstance(self.class_weight, dict):
            if sample_weight is None:
                sample_weight = np.array([self.class_weight.get(label, 1.0) for label in y])
            else:
                sample_weight = sample_weight * np.array([self.class_weight.get(label, 1.0) for label in y])
                
        elif sample_weight is not None:
           sample_weight = np.ones(len(y), dtype="float32")

        self.n_classes_ = len(self.label_encoder.classes_)
        self.classes_ = self.label_encoder.classes_

        return super().fit(X_transformed, y, sample_weight)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """
        Override the _boost method to include a custom loss function for boosting.
        """
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)

        y_pred = estimator.predict(X)

        # Calculate loss using custom or exponential loss function
        if self.loss_function:
            estimator_error = self.loss_function(y, y_pred, sample_weight)
        else:
            incorrect = y_pred != y
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1., 0.

        if estimator_error >= 0.5:
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit.")
            return None, None, None

        # Calculate the estimator weight
        estimator_weight = self.learning_rate * 0.5 * np.log((1. - estimator_error) / max(estimator_error, 1e-10))

        # Update sample weights
        incorrect = y_pred != y
        sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        # Apply the same scaling transformation as in fit
        if self.transformer is not None:
            X_mapped = self.transformer.transform(X)
            X_transformed = self.minmaxscaler.transform(X_mapped)
        else:
            X_transformed = self.minmaxscaler.transform(X)

        y_pred = super().predict(X_transformed)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        # Apply the same scaling transformation as in fit
        if self.transformer is not None:
            X_mapped = self.transformer.transform(X)
            X_transformed = self.minmaxscaler.transform(X_mapped)
        else:
            X_transformed = self.minmaxscaler.transform(X)

        return super().predict_proba(X_transformed)
