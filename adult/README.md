# Basic Support for Adult Dataset
https://archive.ics.uci.edu/ml/datasets/adult

## Requirements
- Python 3.6 >=
- Numpy
- Pandas
- YellowBricks
- scikit-learn
- seaborn

## Notebooks
**logreg_pipeline.ipynb** Logistic regression pipeline
**nb_pipeline.ipynb** Naive Bayes pipeline  
**pipeline_utils.ipynb** Common utility functions to create pipelines. This is run in all the model pipeline notebooks using %run
**ranfor_pipeline.ipynb** Random Forest pipeline training and tuning using randomized search 
**voting_pipeline.ipynb** Voting Classifier ensemble of SVM, MLP, XGB
**xgb_pipeline.ipynb** XGB Pipeline training and tuning using grid search
**error_analysis.ipynb** Code to run error analysis on random forest and xgboost
**exploration.ipynb** Code for data exploration and plots
**Final Plots.ipynb** Code for creating AUC/ classification report plots
**scratch.ipynb** contains code for some of the data exploration and trial code in general


## data_loader
`import data_loader` or `from . import data_loader`
Note: import statement depends on your source root

### data_loader.***load_train_data***(*train_file_path*, *valid_rate*=0.1, *is_df*=True)
**Input**
- ***train_file_path***: training file path of 'adult.data'
- ***valid_rate***: validation data rate (0 - 1), 0.1 by default
- ***is_df***: whether or not returned objects are pandas.DataFrame, True by default

**Output**
- ***train_features***: training feature 2D-array (numpy.ndarray, str and int columns)
- ***train_labels***: training label array (numpy.ndarray)
- ***valid_features***: validation feature 2D-array (numpy.ndarray, str and int columns)
- ***valid_labels***: validation label array (numpy.ndarray)

---
### data_loader.***load_test_data***(*test_file_path*, *is_df*=True)
**Input**
- ***test_file_path***: test file path of 'adult.test'
- ***is_df***: whether or not returned objects are pandas.DataFrame, True by default

**Output**
- ***test_features***: test feature 2D-array (numpy.ndarray, str and int columns)
- ***test_labels***: test label array (numpy.ndarray)
