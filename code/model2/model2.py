#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'TC_AUC',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

op_train = pd.read_csv('../../data/train/operation_train_new.csv')
trans_train = pd.read_csv('../../data/train/transaction_train_new.csv')

op_test = pd.read_csv('../../data/test/test_operation_round2.csv')
trans_test = pd.read_csv('../../data/test/test_transaction_round2.csv')

y = pd.read_csv('../../data/train/tag_train_new.csv')
sub = pd.read_csv('../../data/submit_example.csv')



print('model2, graph building')


#Graph building
op_device_cols = ['device1','device2','device_code1','device_code2','device_code3','mac1','mac2','ip1','wifi','geo_code','ip1_sub']
ts_device_cols = ['device1','device2','device_code1','device_code2','device_code3','mac1','ip1','geo_code','ip1_sub']
ts_acc_cols = ['merchant','acc_id1','acc_id2','acc_id3']

op_train_graph = op_train.copy()
op_test_graph = op_test.copy()
ts_train_graph = trans_train.copy()
ts_test_graph = trans_test.copy()


## UID-device/acc-UID
field = 'UID'
for deliver in op_device_cols:
    cname = field+'_per_'+deliver+'_'
    tmp = op_train.groupby([deliver])[field].agg(['count', 'nunique']).add_prefix(cname).reset_index()
    op_train_graph = op_train_graph.merge(tmp, on = deliver, how = 'left')
    
    tmp = op_test.groupby([deliver])[field].agg(['count', 'nunique']).add_prefix(cname).reset_index()
    op_test_graph = op_test_graph.merge(tmp, on = deliver, how = 'left')


for deliver in ts_device_cols+ts_acc_cols:
    cname = field+'_per_'+deliver+'_'
    tmp = trans_train.groupby([deliver])[field].agg(['count', 'nunique']).add_prefix(cname).reset_index()
    ts_train_graph = ts_train_graph.merge(tmp, on = deliver,how = 'left')
    
    tmp = trans_test.groupby([deliver])[field].agg(['count', 'nunique']).add_prefix(cname).reset_index()
    ts_test_graph = ts_test_graph.merge(tmp, on = deliver,how = 'left')
    

## UID-device-device
for center in op_device_cols:
    for neighbor in op_device_cols:
        cname = neighbor+'_per_'+center
        tmp = op_train.groupby(center)[neighbor].nunique().rename(cname).reset_index()  
        op_train_graph = op_train_graph.merge(tmp, on = center,how = 'left')
        
        tmp = op_test.groupby(center)[neighbor].nunique().rename(cname).reset_index()
        op_test_graph = op_test_graph.merge(tmp, on = center,how = 'left')

for center in ts_device_cols:
    for neighbor in ts_device_cols:
        cname = neighbor+'_per_'+center
        
        tmp = trans_train.groupby(center)[neighbor].nunique().rename(cname).reset_index()
        ts_train_graph = ts_train_graph.merge(tmp, on = center,how = 'left')
        
        tmp = trans_test.groupby(center)[neighbor].nunique().rename(cname).reset_index()
        ts_test_graph = ts_test_graph.merge(tmp, on = center,how = 'left')
        
## UID-acc-acc
for center in ts_acc_cols:
    for neighbor in ts_acc_cols:
        cname = neighbor+'_per_'+center
        
        tmp = trans_train.groupby(center)[neighbor].nunique().rename(cname).reset_index()
        ts_train_graph = ts_train_graph.merge(tmp, on = center,how = 'left')
        
        tmp = trans_test.groupby(center)[neighbor].nunique().rename(cname).reset_index()
        ts_test_graph = ts_test_graph.merge(tmp, on = center,how = 'left')

        
## UID-device-device-UID
for center in op_device_cols:
    for neighbor in op_device_cols:
        if center not in neighbor: 
            cname = 'UID_per_'+center+'_to_'+neighbor
            tmp = op_train.groupby([center,neighbor]).UID.nunique().rename(cname+'_nunique').reset_index()
            op_train_graph = op_train_graph.merge(tmp, on = [center,neighbor],how = 'left')
        
            tmp = op_test.groupby([center,neighbor]).UID.nunique().rename(cname+'_nunique').reset_index()
            op_test_graph = op_test_graph.merge(tmp, on = [center,neighbor],how = 'left')

for center in ts_device_cols:
    for neighbor in ts_device_cols:
        if center not in neighbor: 
            cname = 'UID_per_'+center+'_to_'+neighbor
            tmp = trans_train.groupby([center,neighbor]).UID.nunique().rename(cname+'_nunique').reset_index()
            ts_train_graph = ts_train_graph.merge(tmp, on = [center,neighbor],how = 'left')
        
            tmp = trans_test.groupby([center,neighbor]).UID.nunique().rename(cname+'_nunique').reset_index()
            ts_test_graph = ts_test_graph.merge(tmp, on = [center,neighbor],how = 'left')
            
## UID-acc-acc-UID
for center in ts_acc_cols:
    for neighbor in ts_acc_cols:
        if center not in neighbor: 
            cname = 'UID_per_'+center+'_to_'+neighbor
            tmp = trans_train.groupby([center,neighbor]).UID.nunique().rename(cname+'_nunique').reset_index()
            ts_train_graph = ts_train_graph.merge(tmp, on = [center,neighbor],how = 'left')
        
            tmp = trans_test.groupby([center,neighbor]).UID.nunique().rename(cname+'_nunique').reset_index()
            ts_test_graph = ts_test_graph.merge(tmp, on = [center,neighbor],how = 'left')
            
            
            
#construct features
def get_feature(op,ts,label,df1, df2):
    for field in [c for c in df1.columns if c not in ['ip2','ip2_sub']]:#op.columns[:]:
        if field != 'UID':
            label =label.merge(op.groupby(['UID'])[field].nunique().rename(field+'_nunique').reset_index(),on='UID',how='left')
    #print("1/7")
    
    field = 'UID'
    for deliver in ['device1','device2','device_code1','device_code2','device_code3','mac1','mac2','ip1','wifi','geo_code','ip1_sub']:
        cname = field+'_per_'+deliver+'_'
        tmp = op.drop_duplicates(['UID', deliver]).groupby('UID')[cname+'count'].agg(['sum']).add_prefix(cname+'count_').reset_index()
        label =label.merge(tmp,on='UID',how='left')

        tmp = op.drop_duplicates(['UID', deliver]).groupby('UID')[cname+'nunique'].agg(['sum', 'max','mean']).add_prefix(cname+'nunique_').reset_index()
        label =label.merge(tmp,on='UID',how='left')
        
        label = label.merge(op.groupby([field])[deliver].count().rename(deliver+'_cnt').reset_index(),on='UID',how='left')
    #print('2/7')
    
    for center in ['device1','device2','mac1','mac2','ip1','wifi','geo_code','ip1_sub']:
        for neighbor in ['device1','device2','mac1','mac2','ip1','wifi','geo_code','ip1_sub']:
            if center not in neighbor: 
                cname = neighbor+'_per_'+center
                tmp = op.drop_duplicates(['UID', center]).groupby('UID')[cname].agg(['sum', 'max','mean']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
                cname = 'UID_per_'+center+'_to_'+neighbor
                tmp = op.drop_duplicates(['UID', center, neighbor]).groupby('UID')[cname+'_nunique'].agg(['sum']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
    for center in ['device_code1','device_code2','device_code3']:
        for neighbor in ['device1','device2','mac1','mac2','ip1','wifi','geo_code','ip1_sub']:
            if center not in neighbor: 
                cname = neighbor+'_per_'+center
                tmp = op.drop_duplicates(['UID', center]).groupby('UID')[cname].agg(['sum', 'max','mean']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
                cname = 'UID_per_'+center+'_to_'+neighbor
                tmp = op.drop_duplicates(['UID', center, neighbor]).groupby('UID')[cname+'_nunique'].agg(['sum']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
    for center in ['device1','device2','mac1','mac2','ip1','wifi','geo_code','ip1_sub']:
        for neighbor in ['device_code1','device_code2','device_code3']:
            if center not in neighbor: 
                cname = neighbor+'_per_'+center
                tmp = op.drop_duplicates(['UID', center]).groupby('UID')[cname].agg(['sum', 'max','mean']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
                cname = 'UID_per_'+center+'_to_'+neighbor
                tmp = op.drop_duplicates(['UID', center, neighbor]).groupby('UID')[cname+'_nunique'].agg(['sum']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
    #print('3/7')
    print('op done')   
#-------------------------------------------------------------------------------------------------------------------    
    
    for field in df2.columns[:]:
        if field != 'UID':
            label =label.merge(ts.groupby(['UID'])[field].nunique().reset_index(),on='UID',how='left')
    #print('4/7')
    field = 'UID'
    for deliver in ['device1','device2','device_code1','device_code2','device_code3','mac1','ip1','geo_code','ip1_sub','merchant','acc_id1','acc_id2','acc_id3']:
        
        cname = field+'_per_'+deliver+'_'
        tmp = ts.drop_duplicates(['UID', deliver]).groupby('UID')[cname+'count'].agg(['sum']).add_prefix(cname+'count_').reset_index()
        label =label.merge(tmp,on='UID',how='left')

        tmp = ts.drop_duplicates(['UID', deliver]).groupby('UID')[cname+'nunique'].agg(['sum', 'max','mean']).add_prefix(cname+'_nunique_').reset_index()
        label =label.merge(tmp,on='UID',how='left')
        
        label = label.merge(ts.groupby([field])[deliver].count().rename(deliver+'_cnt').reset_index(),on='UID',how='left')    
    #print('5/7')        
    
    for center in ['device1','device2','mac1','ip1','geo_code','ip1_sub']:
        for neighbor in ['device1','device2','mac1','ip1','geo_code','ip1_sub']:
            if center not in neighbor: 
                cname = neighbor+'_per_'+center
                tmp = ts.drop_duplicates(['UID', center]).groupby('UID')[cname].agg(['sum', 'max','mean']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
                cname = 'UID_per_'+center+'_to_'+neighbor
                tmp = ts.drop_duplicates(['UID', center, neighbor]).groupby('UID')[cname+'_nunique'].agg(['sum']).add_prefix(cname+'_nunique_ts_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
    for center in ['device_code1','device_code2','device_code3']:
        for neighbor in ['device1','device2','mac1','ip1','geo_code','ip1_sub']:
            if center not in neighbor: 
                cname = neighbor+'_per_'+center
                tmp = ts.drop_duplicates(['UID', center]).groupby('UID')[cname].agg(['sum', 'max','mean']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
                cname = 'UID_per_'+center+'_to_'+neighbor
                tmp = ts.drop_duplicates(['UID', center, neighbor]).groupby('UID')[cname+'_nunique'].agg(['sum']).add_prefix(cname+'_nunique_ts_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
    for center in ['device1','device2','mac1','ip1','geo_code','ip1_sub']:
        for neighbor in ['device_code1','device_code2','device_code3']:
            if center not in neighbor: 
                cname = neighbor+'_per_'+center
                tmp = ts.drop_duplicates(['UID', center]).groupby('UID')[cname].agg(['sum', 'max','mean']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left') 
                
                cname = 'UID_per_'+center+'_to_'+neighbor
                tmp = ts.drop_duplicates(['UID', center, neighbor]).groupby('UID')[cname+'_nunique'].agg(['sum']).add_prefix(cname+'_nunique_ts_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
    #print('6/7')
    
    for center in ['merchant','acc_id1','acc_id2','acc_id3']:
        for neighbor in ['merchant','acc_id1','acc_id2','acc_id3']:
            if center not in neighbor: 
                cname = neighbor+'_per_'+center
                tmp = ts.drop_duplicates(['UID', center]).groupby('UID')[cname].agg(['sum', 'max','mean']).add_prefix(cname+'_nunique_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
                cname = 'UID_per_'+center+'_to_'+neighbor
                tmp = ts.drop_duplicates(['UID', center, neighbor]).groupby('UID')[cname+'_nunique'].agg(['sum']).add_prefix(cname+'_nunique_ts_').reset_index()
                label =label.merge(tmp,on='UID',how='left')
                
    #print('7/7')            
    print("ts done")
    return label

print('model2, contruct features')
train = get_feature(op_train_graph,ts_train_graph,y, op_train, trans_train).fillna(-1)
train = train.drop(['Tag'],axis = 1).fillna(-1)
label = y['Tag']
test = get_feature(op_test_graph,ts_test_graph,sub, op_test, trans_test).fillna(-1)
test_id = test['UID']
test = test.drop(['Tag'],axis = 1).fillna(-1)
train = train.drop(['UID'],axis = 1)
test = test.drop(['UID'],axis = 1)



print('model2, model start')
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
    n_estimators=5000, objective='binary', subsample=0.9, colsample_bytree=0.77, subsample_freq=1, learning_rate=0.05,
    random_state=1000, n_jobs=-1, min_child_weight=4, min_child_samples=5, min_split_gain=0)
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test_id.shape[0])

best_score = []
for index, (train_index, test_index) in enumerate(skf.split(train, label)):
    lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
                  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                            (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    
    oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,1]
    
    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
    sub_preds += test_pred / 5
    
    


m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
file_name = int(m[1]*100000)

sub = pd.read_csv("../../data/submit_example.csv")
sub['Tag'] = sub_preds
sub.to_csv('./result/model2_res.csv',index=0)


print('model2,done')

