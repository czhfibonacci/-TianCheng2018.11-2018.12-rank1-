
# coding: utf-8

# In[7]:



# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import geohash
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import ensemble

# import xgboost as xgb
# from xgboost import XGBRegressor
import lightgbm as lgb


# In[2]:


import importlib
import ope_process
importlib.reload(ope_process)
import trans_process
importlib.reload(trans_process)
import ope_new_fea
importlib.reload(ope_new_fea)
import trans_new_fea
importlib.reload(trans_new_fea)
import warnings
warnings.filterwarnings('ignore')




print('model1,start')
# In[2]:


trans_data = trans_process.read_data()
ope_data = ope_process.read_data()


# In[3]:


ope_pop_degree_feature = ope_process.get_ope_pop_degree_feature(ope_data)
ope_latitude_longitude_feature = ope_process.get_ope_latitude_longitude_feature(ope_data)

trans_pop_degree_feature = trans_process.get_trans_pop_degree_feature(trans_data)
trans_latitude_longitude_feature = trans_process.get_trans_latitude_longitude_feature(trans_data)


# In[8]:


#ope_new_feature = ope_new_feature.get_ope_new_feature()
ope_data = ope_new_fea.read_data()
ope_unstack_feature = ope_new_fea.get_ope_unstack_feature(ope_data)
ope_intersection_topK_feature = ope_new_fea.get_ope_intersection_topK_feature()


# In[7]:


ope_feature = pd.concat([ope_unstack_feature, ope_intersection_topK_feature, ope_latitude_longitude_feature, ope_pop_degree_feature], axis=1)

ope_feature['ope_UID_cnt'] = ope_process.get_ope_amount_cnt(ope_data)




# In[9]:




trans_data = trans_new_fea.read_data()
trans_unstack_feature = trans_new_fea.get_trans_unstack_feature(trans_data) #3s
trans_intersection_topK_feature = trans_new_fea.get_trans_intersection_topK_feature() #10s
trans_new_feature = pd.concat([trans_unstack_feature,  trans_intersection_topK_feature], axis=1)


trans_feature = pd.concat([trans_unstack_feature, trans_intersection_topK_feature, trans_latitude_longitude_feature, trans_pop_degree_feature], axis = 1)
trans_feature['trans_UID_cnt'] = trans_process.get_trans_amount_cnt(trans_data)


# In[11]:




feature = pd.concat([ope_feature, trans_feature], axis=1)
feature['trans_combine_ope_UID_cnt_diff'] = feature['ope_UID_cnt'] - feature['trans_UID_cnt']
feature['trans_combine_ope_UID_cnt_ratio'] = (feature['ope_UID_cnt'] + 0.1) / (feature['trans_UID_cnt'] + 0.1)



lgb_params =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'is_training_metric': True,
    'min_data_in_leaf': 50,
    'num_leaves': 16,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    #'bagging_freq':1,
    'verbosity':-1,
    'scale_pos_weight':0.3
    #'feature_fraction_seed': i
}  


label_test = pd.read_csv('../../data/submit_example.csv').drop('Tag', axis = 1)
label_train = pd.read_csv("../../data/train/tag_train_new.csv")
X_train, X_test, y_train = ope_process.feature_matrix_to_train_matrix(feature)
dtrain = lgb.Dataset(X_train,label=y_train)
dtest = lgb.Dataset(X_test)


lgb_model = lgb.train(lgb_params,dtrain,feval=ope_process.lgb_tpr_weight_funtion,verbose_eval=20,num_boost_round=400,valid_sets=[dtrain], early_stopping_rounds=50)
#单模型使用
preds = lgb_model.predict(X_test)

lgb_pred = pd.DataFrame(list(map(lambda x, y: [x, y], label_test['UID'], preds)))
lgb_pred.columns = ['UID', 'Tag']

lgb_pred.to_csv('./result/model1_res.csv',index=False,header=True,sep=',',columns=['UID','Tag'])

print('model1,write result')

print('model1,done')
