#!/usr/bin/env python
# coding: utf-8

# In[15]:


import data_loader
import numpy as np
import pandas


# In[16]:


import warnings
warnings.filterwarnings('ignore')


# In[17]:


# load data and assign names
trdf, valdf = data_loader.load_train_data("data/adult.data", is_df=True)
## adding columns labels https://chartio.com/resources/tutorials/how-to-rename-columns-in-the-pandas-python-library/
trdf.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"
,"target"]
valdf.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"
,"target"]



# # Pipelines
# 
# 

# In[42]:


from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import sklearn.neural_network.multilayer_perceptron as mlp
import xgboost as xgb
from sklearn.externals import joblib

# In[7]:


def select_object(X):
    return X.select_dtypes(include = [np.object])


# In[8]:


def select_number(X):
    return X.select_dtypes(include= [np.int64, np.float64])


# In[9]:


def strip_cols(X):
    return X.apply(lambda col: col.str.strip())


# In[18]:


def target_binarizer(Y):
    lb = LabelBinarizer()
    return lb.fit_transform(Y)


# In[ ]:


# references https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
# references https://scikit-learn.org/stable/auto_examples/preprocessing/plot_function_transformer.html
# references https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py

stringstrip = Pipeline([
                ('selector', FunctionTransformer(select_object,validate=False)),
                ('striper', FunctionTransformer(strip_cols, validate=False))
            ])
numerical_transform = Pipeline([
    ('selector', FunctionTransformer(select_number, validate = False)),
    ('scaler', StandardScaler()) # use get parameters and set parameters for validation set
    ])

cat_transformer = Pipeline([
    ('stringstriper' , stringstrip ),
#     ('missing-indicator', MissingIndicator(missing_values='?', features='all')),
    ('imputer', SimpleImputer(missing_values='?', strategy = 'constant', fill_value='MISSING')),
     ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'))
     ])


# In[20]:


def selectcols(X, name = None):
    return X[name]


# In[30]:


#a = stringstrip.fit_transform(trdf)


# In[31]:


#a['workclass'][1]


# In[32]:


#MissingIndicator()


# In[33]:


#b = numerical_transform.fit_transform(trdf)


# In[34]:


#b


# In[35]:


# cat_transformer.set_params(stringstriper__selector__validate=False)
#cat = cat_transformer.fit_transform(trdf)


# In[36]:


preprocess = FeatureUnion([
    ('numerical_transform', numerical_transform),
    ('cat_transformer', cat_transformer)
])


# In[37]:


pipeline = Pipeline([
    ('preprocess', preprocess),
    ('mlp', mlp.MLPClassifier())
])


# In[38]:


Xtr = trdf.drop('target',axis=1)
Ytr = target_binarizer(trdf['target'])
Xva = valdf.drop('target', axis=1)
Yva = target_binarizer(valdf['target'])

pipeline.fit(Xtr,Ytr)
pipeline.score(Xva, Yva)


# In[40]:

# In[43]:


xgbPipeline = Pipeline([
    ("preprocess", preprocess),
    ("xgb", xgb.XGBClassifier())
])


# In[44]:


xgbPipeline.fit(Xtr, Ytr)


# In[45]:


xgbPipeline.score(Xva,Yva)


# In[46]:


def auc(pipeline, X, Y):
    pred = pipeline.predict(X)
    print(pred)
    fpr, tpr, thresholds = metrics.roc_curve(Y, pred)
    return metrics.auc(fpr, tpr)


# In[47]:


print(auc(xgbPipeline, Xva, Yva))


# In[48]:


print(auc(xgbPipeline, Xtr, Ytr))


# In[49]:


print(auc(pipeline, Xtr, Ytr))


# In[50]:


print(auc(pipeline, Xva, Yva))


# # Hyper-parameter Tuning

# ## Parameter tuning for XGB

# In[51]:

#####################################- GRID SEARCH ON XBG #######################################
# 
# from sklearn.model_selection import GridSearchCV
# hyperparameters = {
#         'xgb__min_child_weight': [1, 5, 10],
#         'xgb__gamma': [0.5, 1, 1.5, 2],
#         'xgb__subsample': [0.6, 0.8, 1.0],
#         'xgb__colsample_bytree': [0.6, 0.8, 1.0],
#         'xgb__max_depth': [3, 4, 5]
#         }
# clf = GridSearchCV(xgbPipeline, hyperparameters, cv=5)
# 
# clf.fit(Xtr, Ytr)
# 
# joblib.dump(clf.best_estimator_, 'xgb_grid_model.pkl')
# joblib.dump(clf.best_params_, 'best_xgb_grid_params.pkl', compress = 1) # Only best parameters

#############################################################################################

# In[75]:

#####################################- GRID SEARCH ON RFC  #######################################

from sklearn.ensemble import RandomForestClassifier
rfPipeline = Pipeline([
    ("preprocess", preprocess),
    ("rfc", RandomForestClassifier())
])


# In[76]:


rfPipeline.fit(Xtr, Ytr.ravel())


# In[77]:


print("rfc score: ",rfPipeline.score(Xva,Yva))
print("auc training score: ", auc(rfPipeline,Xtr,Ytr))
print("auc validation score: ", auc(rfPipeline,Xva,Yva))


# ### Hyper parameterization for RandomForest

# In[ ]:


hyperparameters = {
        'rfc__bootstrap': [True,False],
        'rfc__max_depth': [10,25,50,75,None],
        'rfc__max_features': ['auto','sqrt'],
        'rfc__min_samples_leaf': [1,2,4],
        'rfc__min_samples_split': [2,5,10],
        'rfc__n_estimators':[200,500,1000,1500,2000]
        }
rfc = GridSearchCV(rfPipeline, hyperparameters, cv=10)
rfc.fit(Xtr, Ytr.ravel())


joblib.dump(rfc.best_estimator_, 'rfc_grid_model.pkl')
joblib.dump(rfc.best_params_, 'best_rfc_grid_params.pkl', compress = 1) # Only best parameters
# In[ ]:


#############################################################################################
print(rfc.best_score_)


# In[ ]:


print(rfc.best_estimator_)


# In[ ]:


#xgbPipeline.get_params().keys()


# In[ ]:




