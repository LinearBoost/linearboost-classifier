
LinearBoost Classifier
=======================

LinearBoost is a fast and accurate classification algorithm built to enhance the performance of the linear classifier SEFR. It combines efficiency and accuracy, delivering state-of-the-art F1 scores and classification performance.

In benchmarks across seven well-known datasets, LinearBoost:

- Outperformed XGBoost on all seven datasets
- Surpassed LightGBM on five datasets
- Achieved up to 98% faster runtime compared to both algorithms

Key Features:

- High Accuracy: Comparable to or exceeding Gradient Boosting Decision Trees (GBDTs)
- Exceptional Speed: Blazing fast training and inference times
- Resource Efficient: Low memory usage, ideal for large datasets

## 🚀 New Release (v0.0.2) 


Version 0.0.2 of the **LinearBoost Classifier** is released! This new version introduces several exciting features and improvements:

- 🛠️ Support of custom loss function
- ✅ Enhanced handling of class weights
- 🎨 Customized handling of the data scalers
- ⚡ Optimized boosting
- 🕒 Improved runtime and scalability


Get Started and Documentation
-----------------------------

The documentation is available at https://linearboost.readthedocs.io/.

Results
-------

All of the results are reported based on 10-fold Cross-Validation.

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


Future Developments
-----------------------------
These are not supported in this current version, but are in the future plans:
- Adding a custom loss function
- Supporting class weights
- A replacement for scaling
- Supporting categorical variables
- Adding regression

Reference Paper
-----------------------------
The paper is written by Hamidreza Keshavarz (Independent Researcher based in Berlin, Germany) and Reza Rawassizadeh (Department of Computer Science, Metropolitan college, Boston University, United States). It will be available soon.

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/LinearBoost/linearboost-classifier/blob/main/LICENSE) for additional details.
