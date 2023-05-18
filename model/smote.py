import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('D:/AI_study/team_project/medical_noshow3.csv')
print(df) 

x = df.loc[:, df.columns != 'No-show']
y = df[['No-show']]   

x_train, x_test, y_train,y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

from imblearn.over_sampling import SMOTE

print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', x_train.shape, y_train.shape)
smote = SMOTE(random_state=0)
x_train,y_train = smote.fit_resample(x_train,y_train)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', x_train.shape, y_train.shape)
# SMOTE 적용 후 레이블 값 분포 확인
print('SMOTE 적용 후 레이블 값 분포:')
print(pd.Series(y_train['No-show']).value_counts())

# 변수 설정
n_splits = 11    
random_state = 42
#scaler = StandardScaler()
scaler = MinMaxScaler()


kfold = KFold(n_splits=n_splits, shuffle=True, 
              random_state=random_state)

# Scaler
scaler.fit(x_train)                 
x_train = scaler.transform(x_train)   # train 은 fit, transform 모두 해줘야 함
x_test = scaler.transform(x_test) 

# #parameters
# param = {
#     'learning_rate': [0.1, 0.5, 1], # controls the learning rate
#     'depth': [3, 4, 5], # controls the maximum depth of the tree
#     'l2_leaf_reg': [2, 3, 4], # controls the L2 regularization term on weights
#     'colsample_bylevel': [0.1, 0.2, 0.3], # specifies the fraction of columns to be randomly sampled for each level
#     'n_estimators': [100, 200], # specifies the number of trees to be built
#     'subsample': [0.1, 0.2, 0.3], # specifies the fraction of observations to be randomly sampled for each tree
#     'border_count': [32, 64, 128],# specifies the number of splits for numerical features
#     'bootstrap_type': ['Bernoulli', 'MVS']
# } 

# cat = CatBoostClassifier()
# model = GridSearchCV(cat, param,  cv = kfold, 
#                    refit = True, verbose = 1, n_jobs = -1  )

# model = CatBoostClassifier(bootstrap_type = 'Bernoulli', border_count = 128, 
#                            colsample_bylevel = 0.3, depth = 3, 
#                            l2_leaf_reg = 2, learning_rate =  0.5, 
#                            n_estimators=200, subsample= 0.3) 


# model = CatBoostClassifier( depth = 7, l2_leaf_reg = 7, learning_rate =  0.05, random_state=72) 
# # model_score :  0.9208288092652913


model = CatBoostClassifier( depth = 7, l2_leaf_reg = 7, learning_rate =  0.05, random_state=72) 
# model_score :  0.9208288092652913


# #model = CatBoostClassifier(iterations=3008, depth = 11, learning_rate = 0.8948182120285428, 
#                    random_strength =  9.992204520122275e-05, l2_leaf_reg = 1.3121664107912345, random_state =  1635)


model = CatBoostClassifier( iterations= 1759, depth = 7, l2_leaf_reg = 7,
                           random_strength =  9.992204520122275e-05,
                            learning_rate =  0.8948182120285428, random_state=72) 

# model_score :  0.9647575099529497                           


import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

# print('최적의 파라미터 : ', model.best_params_ )
# print('최적의 매개변수 : ', model.best_estimator_)
# print('best_score : ', model.best_score_)
print('model_score : ', model.score(x_test, y_test))
print('걸린 시간 : ', end_time, '초')


# 최적의 파라미터 :  {'bootstrap_type': 'Bernoulli', 'border_count': 128, 'colsample_bylevel': 0.3, 'depth': 3, 'l2_leaf_reg': 2, 'learning_rate': 0.5, 'n_estimators': 200, 'subsample': 0.3}
# 최적의 매개변수 :  <catboost.core.CatBoostClassifier object at 0x00000201447FD270>
# best_score :  0.9202172008454429
# model_score :  0.9181143684401013
# 걸린 시간 :  19328.9823448658 초

# # SelectFromModel
# from sklearn.feature_selection import SelectFromModel

# thresholds = model.feature_importances_

# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)

#     selection_model = XGBClassifier()
#     selection_model.fit(select_x_train, y_train)
#     y_predict = selection_model.predict(select_x_test)
#     score = accuracy_score(y_test, y_predict)
#     print("Thresh=%.3f, n=%d, ACC:%.2f%%"
#         %(thresh, select_x_train.shape[1], score*100))

