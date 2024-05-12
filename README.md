
LinearBoost Classifier
=======================

LinearBoost is a classification algorithm that is designed to boost a linear classifier algorithm named SEFR. It is an efficient classification algorithm that can result in state-of-the-art accuracy and F1 score. It has the following advantages:

- Fast training speed
- Low memory requirement
- Good accuracy


Get Started and Documentation
-----------------------------

The documentation is available at https://linearboost.readthedocs.io/.

Results
-------

F-Score results on each number of estimators on [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic):

|                |   5 est.   |  10 est.   |  20 est.   |  50 est.   | 100 est.   | 200 est.   | 500 est.   | 1000 est.  |
|----------------|------------|------------|------------|------------|------------|------------|------------|------------|
| XGBoost        | 0.952358   | 0.957749   | 0.954427   | **0.964202**   | 0.964008   | 0.964246   | 0.964246   | 0.964246   |
| LinearBoost (SAMME.R) | 0.926656   | 0.943111   | 0.967024   | 0.967384   | **0.974757**   | 0.962691   | 0.954958   | 0.937239   |
| LinearBoost (SAMME) | 0.960055   | 0.961981   | **0.967724**   | 0.967724   | 0.967724   | 0.967724   | 0.967724   | 0.967724   |
| LightGBM       | 0.931737   | 0.948712   | 0.955928   | 0.963925   | 0.959527   | 0.967475   | **0.971148**   | 0.971148   |
| CatBoost       | 0.945893   | 0.950437   | 0.965537   | 0.969827   | 0.965278   | 0.965639   | **0.971439**   | 0.969537   |

Runtime to achieve the best result:

|                | Time (sec.)|
|----------------|------------|
| XGBoost        | 1.29   |
| LinearBoost (SAMME.R) | 2.24   |
| LinearBoost (SAMME) | 0.51   |
| LightGBM       | 2.79   |
| CatBoost       | 38.25   |

F-Score results on each number of estimators on [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease):

|                |   5 est.   |  10 est.   |  20 est.   |  50 est.   | 100 est.   | 200 est.   | 500 est.   | 1000 est.  |
|----------------|------------|------------|------------|------------|------------|------------|------------|------------|
| XGBoost        | 0.771211   | 0.797882   | 0.798590   | 0.799304   | 0.792604   | 0.792818   | 0.785654   | 0.785643   |
| LinearBoost (SAMME.R) | 0.812511   | 0.831613   | 0.834764   | 0.816657   | 0.793616   | 0.730861   | 0.516908   | 0.365107   |
| LinearBoost (SAMME) | 0.812472   | 0.813964   | 0.814151   | 0.814151   | 0.814151   | 0.814151   | 0.814151   | 0.814151   |
| LightGBM       | 0.817035   | 0.808602   | 0.819666   | 0.812094   | 0.812254   | 0.805578   | 0.795899   | 0.785490   |
| CatBoost       | 0.819977   | 0.832422   | 0.824360   | 0.839461   | 0.839286   | 0.813326   | 0.825896   | 0.829023   |



|                |   5 est.   |  10 est.   |  20 est.   |  50 est.   | 100 est.   | 200 est.   | 500 est.   | 1000 est.  |
|----------------|------------|------------|------------|------------|------------|------------|------------|------------|
| XGBoost        | 0.952358   | 0.957749   | 0.954427   | 0.964202   | 0.964008   | 0.964246   | 0.964246   | 0.964246   |
| LinearBoost (SAMME.R) | 0.926656   | 0.943111   | 0.967024   | 0.967384   | 0.974757   | 0.962691   | 0.954958   | 0.937239   |
| LinearBoost (SAMME) | 0.960055   | 0.961981   | 0.967724   | 0.967724   | 0.967724   | 0.967724   | 0.967724   | 0.967724   |
| LightGBM       | 0.931737   | 0.948712   | 0.955928   | 0.963925   | 0.959527   | 0.967475   | 0.971148   | 0.971148   |
| CatBoost       | 0.945893   | 0.950437   | 0.965537   | 0.969827   | 0.965278   | 0.965639   | 0.971439   | 0.969537   |


Reference Paper
-----------------------------
The paper is written by Hamidreza Keshavarz (Independent Researcher based in Berlin) and Reza Rawassizadeh (Department of Computer Science, Metropolitan college, Boston University, United States). It will be available soon.

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/microsoft/LightGBM/blob/master/LICENSE) for additional details.
