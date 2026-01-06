# LinearBoost Classifier

![Lastest Release](https://img.shields.io/badge/release-v0.1.5-green)
[![PyPI Version](https://img.shields.io/pypi/v/linearboost)](https://pypi.org/project/linearboost/)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![PyPI Downloads](https://static.pepy.tech/badge/linearboost)](https://pepy.tech/projects/linearboost)

## 🧪 Quickstart Demo

Want to see how LinearBoost works in practice?

➡️ [**Run the demo notebook**](notebooks/demo_linearboost_usage.ipynb)

This Jupyter notebook shows how to:
- Load and prepare data
- Train `LinearBoostClassifier`
- Evaluate using F1 score and cross-validation

LinearBoost is a fast and accurate classification algorithm built to enhance the performance of the linear classifier SEFR. It combines efficiency and accuracy, delivering state-of-the-art F1 scores and classification performance.

In benchmarks across seven well-known datasets, LinearBoost:

- Outperformed XGBoost on all seven datasets
- Surpassed LightGBM on five datasets
- Achieved up to **98% faster runtime** compared to both algorithms

Key Features:

- High Accuracy: Comparable to or exceeding Gradient Boosting Decision Trees (GBDTs)
- Exceptional Speed: Blazing fast training and inference times
- Resource Efficient: Low memory usage, ideal for large datasets

---

## 🚀 New in Version 0.1.5

The latest release introduces major architectural improvements designed for **scalability**, **robustness on imbalanced data**, and **training speed**.

### ⚡ Scalable Kernel Approximation

LinearBoost now supports **Kernel Approximation** via `kernel_approx='rff'` or `kernel_approx='nystrom'`.

**Why it matters:** Previously, non-linear kernels required computing a full \(O(n^2)\) kernel matrix, which is memory-intensive for large datasets.

**New Capability:** You can now map inputs to a lower-dimensional feature space using:
- **Random Fourier Features (RFF)** — for RBF kernels
- **Nyström Approximation** — for any kernel type

This enables linear time complexity while retaining non-linear decision boundaries.

```python
# Example: Using kernel approximation for scalable non-linear classification
clf = LinearBoostClassifier(
    kernel='rbf',
    kernel_approx='rff',  # or 'nystrom'
    n_components=256
)
```

### 🎯 Stochastic Boosting & Regularization

Advanced regularization techniques to prevent overfitting and reduce variance:

- **Subsampling (`subsample`)**: Enables Stochastic Gradient Boosting by training each estimator on a random fraction of the training data.
- **Shrinkage (`shrinkage`)**: Scales the contribution of each new estimator (learning rate decay), effectively "slowing down" learning for better generalization.

```python
clf = LinearBoostClassifier(
    subsample=0.8,    # Use 80% of data per iteration
    shrinkage=0.9     # Scale each estimator's contribution by 0.9
)
```

### ⚖️ Optimized for Imbalanced Data

The internal boosting logic has been overhauled to prioritize **F1-Score optimization**:

- **Adaptive Class Weighting**: The algorithm dynamically adjusts sample weights based on class frequencies within the boosting loop, aggressively correcting errors on minority classes.
- **F1-Based Estimator Weighting**: Estimators are rewarded not just for accuracy, but specifically for their F1 performance.

### ⏱️ Early Stopping

Training can now stop automatically when validation scores plateau:

- **Standard validation splits** via `validation_fraction`
- **Out-of-Bag (OOB) Evaluation**: When using subsampling (`subsample < 1.0`), LinearBoost utilizes unused samples for validation without reducing training set size.

```python
clf = LinearBoostClassifier(
    n_estimators=500,
    early_stopping=True,
    validation_fraction=0.1,  # 10% held out for validation
    n_iter_no_change=5,       # Stop after 5 iterations with no improvement
    tol=1e-4
)

# Or with OOB evaluation (automatic when subsampling)
clf = LinearBoostClassifier(
    n_estimators=500,
    subsample=0.8,            # Enables OOB evaluation
    early_stopping=True,
    n_iter_no_change=5
)
```

---

## 🚀 New Major Release (v0.1.3)
The `LinearBoost` and `SEFR` classifiers use kernels to solve non-linear problems. Kernels work by projecting data into a different perspective, allowing a simple linear model to capture complex, curved patterns.

---

### Linear Kernel

This is the default setting and performs no transformation. It's the fastest option and works best when the data is already simple and doesn't require complex boundaries.

---

### Polynomial Kernel

This kernel creates feature interactions to model curved or complex relationships in the data.

* **`degree`**: Sets the complexity of the feature combinations.
* **`coef0`**: Balances the influence between high-degree and low-degree feature interactions.

---

### RBF Kernel

A powerful and flexible kernel that can handle very complex boundaries. It works by considering the distance between data points, making it a strong general-purpose choice.

* **`gamma`**: Controls the reach of a single training point's influence. Small values create smoother boundaries, while large values create more complex ones.

---

### Sigmoid Kernel

Inspired by neural networks, this kernel is useful for certain classification tasks that follow a sigmoid shape.

* **`gamma`**: Scales the data's influence.
* **`coef0`**: Shifts the decision boundary.

## 🚀 New Major Release (v0.1.2)
Version 0.1.2 of **LinearBoost Classifier** is released. Here are the changes:

- The codebase is refactored into a new structure.
- SAMME.R algorithm is returned to the classifier.
- Both SEFR and LinearBoostClassifier classes are refactored to fully adhere to Scikit-learn's conventions and API. Now, they are standard Scikit-learn estimators that can be used in Scikit-learn pipelines, grid search, etc.
- Added unit tests (using pytest) to ensure the estimators adhere to Scikit-learn conventions.
- Added fit_intercept parameter to SEFR similar to other linear estimators in Scikit-learn (e.g., LogisticRegression, LinearRegression, etc.).
- Removed random_state parameter from LinearBoostClassifier as it doesn't affect the result, since SEFR doesn't expose a random_state argument. According to Scikit-learn documentation for this parameter in AdaBoostClassifier:
  > it is only used when estimator exposes a random_state.
- Added docstring to both SEFR and LinearBoostClassifier classes.
- Used uv for project and package management.
- Used ruff and isort for formatting and lining.
- Added a GitHub workflow (*.github/workflows/ci.yml*) for CI on PRs.
- Improved Scikit-learn compatibility.


Get Started and Documentation
-----------------------------

The documentation is available at https://linearboost.readthedocs.io/.

## Recommended Parameters for LinearBoost

The following parameters yielded optimal results during testing. All results are based on 10-fold Cross-Validation:

- **`n_estimators`**:
  A range of 10 to 200 is suggested, with higher values potentially improving performance at the cost of longer training times. When using `early_stopping=True`, you can set a higher value (e.g., 500) and let training stop automatically.

- **`learning_rate`**:
  Values between 0.01 and 1 typically perform well. Adjust based on the dataset's complexity and noise.

- **`algorithm`**:
  Use either `SAMME` or `SAMME.R`. The choice depends on the specific problem:
  - `SAMME`: May be better for datasets with clearer separations between classes.
  - `SAMME.R`: Can handle more nuanced class probabilities.

  **Note:** As of scikit-learn v1.6, the `algorithm` parameter is deprecated and will be removed in v1.8. LinearBoostClassifier will only implement the 'SAMME' algorithm in newer versions.

- **`scaler`**:
  The following scaling methods are recommended based on dataset characteristics:
  - `minmax`: Best for datasets where features are on different scales but bounded.
  - `robust`: Effective for datasets with outliers.
  - `quantile-uniform`: Normalizes features to a uniform distribution.
  - `quantile-normal`: Normalizes features to a normal (Gaussian) distribution.

- **`kernel`** *(new in v0.1.3)*:
  Choose based on data complexity:
  - `linear`: Fastest, for linearly separable data.
  - `rbf`: Most flexible, works well for complex non-linear patterns.
  - `poly`: For polynomial relationships.
  - `sigmoid`: For sigmoid-like decision boundaries.

- **`kernel_approx`** *(new in v0.1.5)*:
  For large datasets with non-linear kernels:
  - `None`: Use full kernel matrix (default, exact but \(O(n^2)\) memory).
  - `'rff'`: Random Fourier Features (only with `kernel='rbf'`).
  - `'nystrom'`: Nyström approximation (works with any kernel).

- **`subsample`** *(new in v0.1.5)*:
  Values in (0, 1] control stochastic boosting. Use `0.8` for variance reduction while maintaining speed.

- **`shrinkage`** *(new in v0.1.5)*:
  Values in (0, 1] scale each estimator's contribution. Use `0.8-0.95` to improve generalization.

- **`early_stopping`** *(new in v0.1.5)*:
  Set to `True` with `n_iter_no_change=5` and `tol=1e-4` to automatically stop training when validation performance plateaus.

These parameters should serve as a solid starting point for most datasets. For fine-tuning, consider using hyperparameter optimization tools like [Optuna](https://optuna.org/).

Results
-------

All of the results are reported based on 10-fold Cross-Validation. The weighted F1 score is reported, i.e. f1_score(y_valid, y_pred, average = 'weighted').

## Performance Comparison: F1 Scores Across Datasets

The following table presents the F1 scores of LinearBoost in comparison with XGBoost, CatBoost, and LightGBM across seven standard benchmark datasets. **Each result is obtained by running Optuna with 200 trials** to find the best hyperparameters for each algorithm and dataset, ensuring a fair and robust comparison.

| Dataset                              | XGBoost  | CatBoost | LightGBM | LinearBoost |
|--------------------------------------|----------|----------|----------|-------------|
| Breast Cancer Wisconsin (Diagnostic) | 0.9767   | 0.9859   | 0.9771   | 0.9822      |
| Heart Disease                        | 0.8502   | 0.8529   | 0.8467   | 0.8507      |
| Pima Indians Diabetes Database       | 0.7719   | 0.7776   | 0.7816   | 0.7753      |
| Banknote Authentication              | 0.9985   | 1.0000   | 0.9993   | 1.0000      |
| Haberman's Survival                  | 0.7193   | 0.7427   | 0.7257   | 0.7485      |
| Loan Status Prediction               | 0.8281   | 0.8495   | 0.8277   | 0.8387      |
| PCMAC                                | 0.9310   | 0.9351   | 0.9361   | 0.9331      |

### Experiment Details
- **Hyperparameter Optimization**:
  - Each algorithm was tuned using **Optuna**, a powerful hyperparameter optimization framework.
  - **200 trials** were conducted for each algorithm-dataset pair to identify the optimal hyperparameters.
- **Consistency**: This rigorous approach ensures fair comparison by evaluating each algorithm under its best-performing configuration.

### Key Highlights
- **LinearBoost** achieves competitive or superior F1 scores compared to the state-of-the-art algorithms.
- **Haberman's Survival**: LinearBoost achieves the highest F1 score (**0.7485**), outperforming all other algorithms.
- **Banknote Authentication**: LinearBoost matches the perfect F1 score of **1** achieved by CatBoost.
- LinearBoost demonstrates consistent performance across diverse datasets, making it a robust and efficient choice for classification tasks.

## Runtime Comparison: Time to Reach Best F1 Score

The following table shows the runtime (in seconds) required by LinearBoost, XGBoost, CatBoost, and LightGBM to achieve their best F1 scores. **Each result is obtained by running Optuna with 200 trials** to optimize the hyperparameters for each algorithm and dataset.

| Dataset                              | XGBoost  | CatBoost | LightGBM | LinearBoost |
|--------------------------------------|----------|----------|----------|-------------|
| Breast Cancer Wisconsin (Diagnostic) | 3.22     | 9.68     | 4.52     | 0.30        |
| Heart Disease                        | 1.13     | 0.60     | 0.51     | 0.49        |
| Pima Indians Diabetes Database       | 6.86     | 3.50     | 2.52     | 0.16        |
| Banknote Authentication              | 0.46     | 4.26     | 5.54     | 0.33        |
| Haberman's Survival                  | 4.41     | 8.28     | 5.72     | 0.11        |
| Loan Status Prediction               | 0.83     | 97.89    | 28.41    | 0.44        |
| PCMAC                                | 150.33   | 83.52    | 42.23    | 75.06       |

### Experiment Details
- **Hyperparameter Optimization**:
  - Each algorithm was tuned using **Optuna** with **200 trials** per algorithm-dataset pair.
  - The runtime includes the time to reach the best F1 score using the optimized hyperparameters.
- **Fair Comparison**: All algorithms were evaluated under their best configurations to ensure consistency.

### Key Highlights
- **LinearBoost** demonstrates exceptional runtime efficiency while achieving competitive F1 scores:
  - **Breast Cancer Wisconsin (Diagnostic)**: LinearBoost achieves the best F1 score in just **0.30 seconds**, compared to **3.22 seconds** for XGBoost and **9.68 seconds** for CatBoost.
  - **Loan Status Prediction**: LinearBoost runs in **0.44 seconds**, outperforming LightGBM (**28.41 seconds**) and CatBoost (**97.89 seconds**).
- Across most datasets, LinearBoost reduces runtime by up to **98%** compared to XGBoost and LightGBM while maintaining competitive performance.

### Tuned Hyperparameters

#### XGBoost
```python
params = {
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'n_estimators': trial.suggest_int('n_estimators', 20, 1000),
    'max_depth': trial.suggest_int('max_depth', 1, 20),
    'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.7),
    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
    'enable_categorical': True,
    'eval_metric': 'logloss'
}
```

#### CatBoost
```python
params = {
    'iterations': trial.suggest_int('iterations', 50, 500),
    'depth': trial.suggest_int('depth', 1, 16),
    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.5),
    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
    'random_strength': trial.suggest_loguniform('random_strength', 1e-8, 10.0),
    'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 1e-1, 10.0),
    'border_count': trial.suggest_int('border_count', 32, 255),
    'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
    'rsm': trial.suggest_uniform('rsm', 0.1, 1.0),
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
    'cat_features': categorical_cols
}
```

#### LightGBM
```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
    'n_estimators': trial.suggest_int('n_estimators', 20, 1000),
    'max_depth': trial.suggest_int('max_depth', 1, 20),
    'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
    'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
    'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-8, 1.0),
    'cat_smooth': trial.suggest_int('cat_smooth', 1, 100),
    'cat_l2': trial.suggest_loguniform('cat_l2', 1e-8, 10.0),
    'verbosity': -1
}
```

#### LinearBoost
```python
params = {
    'n_estimators': trial.suggest_int('n_estimators', 10, 500),
    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1),
    'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R']),
    'scaler': trial.suggest_categorical('scaler', ['minmax', 'robust', 'quantile-uniform', 'quantile-normal']),
    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'shrinkage': trial.suggest_float('shrinkage', 0.7, 1.0),
    'early_stopping': True,
    'n_iter_no_change': 5,
}
```

### Why LinearBoost?
LinearBoost's combination of **runtime efficiency** and **high accuracy** makes it a powerful choice for real-world machine learning tasks, particularly in resource-constrained or real-time applications.

### 📰 Featured in:
- [LightGBM Alternatives: A Comprehensive Comparison](https://nightwatcherai.com/blog/lightgbm-alternatives)
  _by Jordan Cole, March 11, 2025_
  *Discusses how LinearBoost outperforms traditional boosting frameworks in terms of speed while maintaining accuracy.*


Future Developments
-----------------------------
These are not yet supported in this current version, but are in the future plans:
- Supporting categorical variables natively
- Adding regression support (`LinearBoostRegressor`)
- Multi-output classification

Reference Paper
-----------------------------
The paper is written by Hamidreza Keshavarz (Independent Researcher based in Berlin, Germany) and Reza Rawassizadeh (Department of Computer Science, Metropolitan college, Boston University, United States). It will be available soon.

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/LinearBoost/linearboost-classifier/blob/main/LICENSE) for additional details.

## Acknowledgments

Some portions of this code are adapted from the scikit-learn project
(https://scikit-learn.org), which is licensed under the BSD 3-Clause License.
See the `licenses/` folder for details. The modifications and additions made to the original code are licensed under the MIT License © 2025 Hamidreza Keshavarz, Reza Rawassizadeh.
The original code from scikit-learn is available at [scikit-learn GitHub repository](https://github.com/scikit-learn/scikit-learn)

Special Thanks to:
- **Mehdi Samsami** – for software engineering, refactoring, and ensuring compatibility.
