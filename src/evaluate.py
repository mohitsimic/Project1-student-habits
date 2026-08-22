import pandas as pd 
import numpy as np


data = pd.read_csv('/Users/mohitsimic/Desktop/project1-student-habits/data/student_habits_performance.csv')
X, y = data.drop(columns=['exam_score', 'student_id']), data['exam_score']

from sklearn.model_selection import (
    train_test_split, cross_validate, cross_val_score
    )
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

train_X, test_X , train_y, test_y = train_test_split(
    X,y,random_state=42, shuffle= True, test_size= .3
)
categorical_head = X.select_dtypes(include = ['str']).columns.tolist()

for i in range(len(categorical_head)):
    col = categorical_head[i]
    X[col] = X[col].astype('category')
numerical_cols = X.select_dtypes([float, int]).columns.tolist()
categorical_data = X.select_dtypes(['category']).columns.tolist()

    
    
cat_transformer = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                                OneHotEncoder(handle_unknown='ignore'))

num_transformer = make_pipeline(SimpleImputer(strategy ='mean'),
                                StandardScaler())



transformer  = ColumnTransformer([('category',
                cat_transformer,categorical_data),
            ('numerical', num_transformer ,numerical_cols)])


from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                             root_mean_squared_error, max_error,r2_score)

##----I made a pipeline so that the test_X can be cleaned to before predictions.

model = make_pipeline(transformer,RandomForestRegressor(random_state=42))
model.fit(train_X,train_y)

predictions = (model.predict(test_X))
kf = KFold(n_splits= 7, shuffle=True, random_state=42)
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error',
           'neg_max_error','neg_root_mean_squared_error','r2']
scores = cross_validate(model, train_X, train_y, cv = kf,scoring= scoring)
print('----For the RandomForest----')
print(f'RMSE : {-scores['test_neg_root_mean_squared_error'].mean()}')
print(f'r2 : {scores['test_r2'].mean()}')
print('\n')
importances = model.named_steps['randomforestregressor'].feature_importances_
feature_names = model.named_steps['columntransformer'].get_feature_names_out()

import pandas as pd

importance_df = pd.Series(importances, index = feature_names).sort_values(ascending=False)

##-- DummyRegressor Results ----

from sklearn.dummy import DummyRegressor
model_2 = DummyRegressor()
model_2.fit(train_X,train_y)
predict = model_2.predict(test_X)
print('----Dummy Regressor Results----')
print(f'RMSE : {root_mean_squared_error(test_y,predict)}')
print(f'r2 : {r2_score(test_y, predict)}')
print('\n')
from sklearn.ensemble import GradientBoostingRegressor

import numpy as np


##---GradientBoosting Results ------

grad_model = make_pipeline(transformer,GradientBoostingRegressor(
    random_state= 42
))
kf_grad = KFold(n_splits= 7, shuffle=True, random_state=42)
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error',
           'neg_max_error','neg_root_mean_squared_error','r2']
scores_grad = cross_validate(grad_model, train_X, train_y, cv = kf,scoring= scoring)
print('----for Gradiet_Boosting----')
print(f'RMSE: {-scores_grad['test_neg_root_mean_squared_error'].mean()}')
print(f'r2 : {scores_grad['test_r2'].mean()}')

print('\n')

##---Linear Model results 

from sklearn.linear_model import LinearRegression
lin_model = make_pipeline(transformer , LinearRegression())
lin_model.fit(train_X,train_y)
kf_linear = KFold(n_splits= 7, shuffle=True, random_state=42)
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error',
           'neg_max_error','neg_root_mean_squared_error','r2']
scores_linear = cross_validate(lin_model, train_X, train_y, cv = kf,scoring= scoring)
print('----For Linear Regression----')
print(f'RMSE : {-scores_linear['test_neg_root_mean_squared_error'].mean()}')
print(f'r2 : {scores_linear['test_r2'].mean()}')