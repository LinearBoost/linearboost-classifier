
LinearBoost Classifier
=======================

LinearBoost is a classification algorithm that is designed to boost a linear classifier algorithm named SEFR. It is an efficient classification algorithm that can result in state-of-the-art accuracy and F1 score. It has the following advantages:

- Fast training speed
- Low memory footprint
- Accuracy on par with Gradient Boosting Decision Trees


Get Started and Documentation
-----------------------------

The documentation is available at https://linearboost.readthedocs.io/.

Results
-------

** All of the results are reported based on 10-fold Cross-Validation. **


F-Score results on each number of estimators on [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic):

| Method         |   5 est.   |  10 est.   |  20 est.   |  50 est.   | 100 est.   | 200 est.   | 500 est.   | 1000 est.  |
|----------------|------------|------------|------------|------------|------------|------------|------------|------------|
| XGBoost        | 0.952358   | 0.957749   | 0.954427   | **0.964202**   | 0.964008   | 0.964246   | 0.964246   | 0.964246   |
| LightGBM       | 0.931737   | 0.948712   | 0.955928   | 0.963925   | 0.959527   | 0.967475   | **0.971148**   | 0.971148   |
| CatBoost       | 0.945893   | 0.950437   | 0.965537   | 0.969827   | 0.965278   | 0.965639   | **0.971439**   | 0.969537   |
| LinearBoost (SAMME.R) | 0.926656   | 0.943111   | 0.967024   | 0.967384   | **0.974757**   | 0.962691   | 0.954958   | 0.937239   |
| LinearBoost (SAMME) | 0.960055   | 0.961981   | **0.967724**   | 0.967724   | 0.967724   | 0.967724   | 0.967724   | 0.967724   |

Runtime to achieve the best result:

| Method         | Time (sec.)|
|----------------|------------|
| XGBoost        | 1.29   |
| LightGBM       | 2.79   |
| CatBoost       | 38.25   |
| LinearBoost (SAMME.R) | 2.24   |
| LinearBoost (SAMME) | 0.51   |

F-Score results on each number of estimators on [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease):

| Method         |   5 est.   |  10 est.   |  20 est.   |  50 est.   | 100 est.   | 200 est.   | 500 est.   | 1000 est.  |
|----------------|------------|------------|------------|------------|------------|------------|------------|------------|
| XGBoost        | 0.771211   | 0.797882   | 0.798590   | **0.799304**   | 0.792604   | 0.792818   | 0.785654   | 0.785643   |
| LightGBM       | 0.817035   | 0.808602   | **0.819666**   | 0.812094   | 0.812254   | 0.805578   | 0.795899   | 0.785490   |
| CatBoost       | 0.819977   | 0.832422   | 0.824360   | **0.839461**   | 0.839286   | 0.813326   | 0.825896   | 0.829023   |
| LinearBoost (SAMME.R) | 0.812511   | 0.831613   | **0.834764**   | 0.816657   | 0.793616   | 0.730861   | 0.516908   | 0.365107   |
| LinearBoost (SAMME) | 0.812472   | 0.813964   | **0.814151**   | 0.814151   | 0.814151   | 0.814151   | 0.814151   | 0.814151   |


Runtime to achieve the best result:

| Method         | Time (sec.)|
|----------------|------------|
| XGBoost        | 0.44   |
| LightGBM       | 0.19   |
| CatBoost       | 0.96   |
| LinearBoost (SAMME.R) | 0.28   |
| LinearBoost (SAMME) | 0.19   |

F-Score results on each number of estimators on [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data):

| Method         |   5 est.   |  10 est.   |  20 est.   |  50 est.   | 100 est.   | 200 est.   | 500 est.   | 1000 est.  |
|----------------|------------|------------|------------|------------|------------|------------|------------|------------|
| XGBoost      | 0.650576   | 0.668125   | 0.654738   | 0.665422   | 0.673953   | 0.675264   | **0.685577**   | 0.679165   |
| LightGBM      | 0.465204   | 0.599001   | 0.666242   | 0.672557   | **0.675394**   | 0.672356   | 0.652203   | 0.637698   |
| CatBoost      | 0.623644   | 0.633344   | 0.663266   | 0.647885   | 0.669377   | 0.660652   | 0.657485   | **0.671585**   |
| LinearBoost (SAMME.R)      | 0.690282   | **0.697498**   | 0.685841   | 0.622432   | 0.461522   | 0.411345   | 0.411345   | 0.411345   |
| LinearBoost (SAMME)      | 0.676735   | 0.681165   | **0.683737**   | 0.683737   | 0.683737   | 0.683737   | 0.683737   | 0.683737   |

Runtime to achieve the best result:

| Method         | Time (sec.)|
|----------------|------------|
| XGBoost        | 4.18   |
| LightGBM       | 1.14   |
| CatBoost       | 192.03   |
| LinearBoost (SAMME.R) | 0.81   |
| LinearBoost (SAMME) | 0.83   |

F-Score results on each number of estimators on [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators):


| Method         |   5 est.   |  10 est.   |  20 est.   |  50 est.   | 100 est.   | 200 est.   | 500 est.   | 1000 est.  |
|----------------|------------|------------|------------|------------|------------|------------|------------|------------|
| XGBoost      | 0.526730   | 0.562816   | 0.587322   | 0.592467   | 0.593964   | 0.594074   | 0.598566   | **0.603016**   |
| LightGBM      | 0.462557   | 0.462557   | 0.529107   | 0.580976   | 0.588251   | 0.590069   | 0.591296   | **0.591785**   |
| CatBoost      | 0.570664   | 0.584894   | 0.590143   | 0.590830   | 0.592464   | **0.593707**   | 0.592682   | 0.592633   |
| LinearBoost (SAMME.R)      | 0.652007   | 0.661966   | **0.663046**   | 0.592903   | 0.469198   | 0.462557   | 0.462557   | 0.462557   |
| LinearBoost (SAMME)      | **0.637149**   | 0.637149   | 0.637149   | 0.637149   | 0.637149   | 0.637149   | 0.637149   | 0.637149   |

Runtime to achieve the best result:

| Method         | Time (sec.)|
|----------------|------------|
| XGBoost        | 395.36   |
| LightGBM       | 307.72   |
| CatBoost       | 192.80   |
| LinearBoost (SAMME.R) | 221.21   |
| LinearBoost (SAMME) | 12.42   |


F-Score results on each number of estimators on [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download):


| Method         |   5 est.   |  10 est.   |  20 est.   |  50 est.   | 100 est.   | 200 est.   | 500 est.   | 1000 est.  |
|----------------|------------|------------|------------|------------|------------|------------|------------|------------|
| XGBoost      | 0.487303   | 0.490594   | 0.511990   | 0.531004   | 0.536601   | **0.538732**   | 0.535431   | 0.534107   |
| LightGBM      | 0.487509   | 0.491262   | 0.498842   | 0.504951   | 0.513150   | 0.517687   | **0.521545**   | 0.520001   |
| CatBoost      | 0.487509   | 0.497890   | 0.516388   | 0.524389   | 0.529016   | 0.519215   | 0.522405   | **0.531045**   |
| LinearBoost (SAMME.R)      | 0.544310   | 0.553557   | 0.565717   | **0.596718**   | 0.491107   | 0.487509   | 0.487509   | 0.487509   |
| LinearBoost (SAMME)      | 0.553043   | **0.570221**   | 0.570013   | 0.570013   | 0.570013   | 0.570013   | 0.570013   | 0.570013   |

Runtime to achieve the best result:

| Method         | Time (sec.)|
|----------------|------------|
| XGBoost        | 2.33   |
| LightGBM       | 6.26   |
| CatBoost       | 159.26   |
| LinearBoost (SAMME.R) | 3.58   |
| LinearBoost (SAMME) | 0.86   |

Reference Paper
-----------------------------
The paper is written by Hamidreza Keshavarz (Independent Researcher based in Berlin, Germany) and Reza Rawassizadeh (Department of Computer Science, Metropolitan college, Boston University, United States). It will be available soon.

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/microsoft/LightGBM/blob/master/LICENSE) for additional details.
