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



def lgb_tpr_weight_funtion(preds,dtrain):
    d = pd.DataFrame()
    d['prob'] = preds
    d['y'] = dtrain.get_label()
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = sum(d['y'])
    NegAll = len(d['y']) - PosAll
    #PosAll = pd.Series(y).value_counts()[1]
    #NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    #return "res", TR1,True
    return "res", 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3, True



#小时映射表
hour_bin = {
    0:0,
    1:0,
    2:0,
    3:0,
    4:0,
    5:0,
    6:0,
    7:1,
    8:1,
    9:1,
    10:1,
    11:1,
    12:3,
    13:3,
    14:4,
    15:4,
    16:4,
    17:4,
    18:5,
    19:5,
    20:5,
    21:5,
    22:5,
    23:0
}


def get_trans_amount_cnt(trans_data, primary_key = 'UID'):
    return trans_data.groupby([primary_key])[primary_key].count()




def trans_TopKOneHot_with_static(df, col_name, k = -1, primary_key = 'UID', fillna = -1, prefix = "topK_one_hot_", topK_keys = []):
    
    #如何选取K 是一个很大的问题
    if (k == -1):
        k = int(len(df[col_name].value_counts()) * 0.2)
    if (len(topK_keys) == 0):
        topK_keys = list(df[col_name].value_counts().index)[:k]
    tmp_col = df[col_name].copy()
    df[col_name] = df[col_name].apply(lambda x : x if (x in topK_keys) else 'THE_OTHER')
    
    topK_one_hot = df.groupby(['UID', col_name])[col_name].count().unstack(col_name).add_prefix(prefix + col_name + '_unstack_')
    topK_static = df.groupby(['UID', col_name])[col_name].count().groupby(['UID']).agg(
                                                    ['mean', 'max', 'min', np.ptp]).add_prefix(prefix + col_name + '_static_')
    topK_one_hot.fillna(0, inplace=True)
    #计算other列和NAN列所占总数的比值
    trans_UID_cnt = get_trans_amount_cnt(df)
    topK_one_hot[prefix + col_name +  '_THE_OTHER_ratio'] = topK_one_hot[prefix + col_name + '_unstack_THE_OTHER'] / trans_UID_cnt
    topK_one_hot[prefix + col_name + '_TopK_ratio'] = 1 - topK_one_hot[prefix + col_name + '_THE_OTHER_ratio']
    df[col_name] = tmp_col.copy()
    topK_feature = pd.concat([topK_one_hot, topK_static], axis = 1)
    return topK_feature


def get_trans_unstack_feature(trans_data):
    #unstack feature
    trans_unstack_col = [
        'channel', 
        'trans_hour',
        'hour_bin',
        'amt_src1',
        'trans_type1', ##有几个类别是不同的
        'is_strange_mac1', 
        'is_strange_bal',
        'trans_type2',
        'market_type'
    ]
    trans_unstack_feature = []
    for col in trans_unstack_col:
        trans_unstack_feature.append(trans_data.groupby(['UID', col])[col].count().unstack(col).add_prefix('trans_' + col + '_unstack_'))
        trans_unstack_feature.append(trans_data.groupby(['UID', col])[col].count().groupby(['UID']).agg(
                                                        ['mean', 'std', 'max', 'min', 'median', np.ptp]).add_prefix('trans_' + col + '_static_'))
    trans_unstack_feature = pd.concat(trans_unstack_feature, axis = 1)
    return trans_unstack_feature    
    
def get_trans_topK_feature(trans_data):
    #top_K feature
    trans_topK_col = [
        'trans_amt',
        'merchant',
        'device1',
        'device2',
        'trans_device_main_kind',
        'amt_src2',
    ]
    trans_topK_k_value = {
        'trans_amt': 25,
        'merchant': 10,
        'device1': 20,
        'device2': 35,
        'trans_device_main_kind': 25,#这两个字段k感觉稍微大了点
        'amt_src2':25
    }
    trans_topK_feature = []
    for col in trans_topK_col:
        trans_topK_feature.append(trans_TopKOneHot_with_static(trans_data, col_name = col, k = trans_topK_k_value[col],prefix = 'trans_'))
    trans_topK_feature = pd.concat(trans_topK_feature, axis = 1)
    return trans_topK_feature



#交集取测试集相交的前k个
def get_trans_intersection_topK_feature():
    

    #topK_intersection_feature
    trans_topK_intersection_col = [
        'merchant',
        'bal',
        'ip1_sub'
    ]

    trans_topK_intersection_k_value = {
        'merchant': 25,
        'bal': 10,
        'ip1_sub': 15
    }
    train_data = pd.read_csv("../../data/train/transaction_train_new.csv", 
                        dtype = {
                            'device1': str,
                            'geo_code':str
                        })
    test_data = pd.read_csv("../../data/test/transaction_round1_new.csv",
                        dtype = {
                            'device1': str,
                            'geo_code':str
                        })
    trans_data = pd.concat([train_data, test_data])
    topK_names = ['merchant', 'code1', 'code2', 'acc_id1', 'device_code1', 'device_code2', 'device_code3',
                  'device1', 'device2', 'mac1', 'ip1', 'acc_id2', 'acc_id3', 'market_code', 'ip1_sub']
    topK_feature = []
    for col in trans_topK_intersection_col:
        k = trans_topK_intersection_k_value[col]
        target_list = train_data[col].value_counts().index
        topK_keys = [name for name in test_data[col].value_counts().index if name in target_list]
        if (len(topK_keys) > k):
            topK_keys = topK_keys[:k]
        topK_feature.append(trans_TopKOneHot_with_static(trans_data, col,primary_key = 'UID', prefix = "trans_intersection_" + col + '_', topK_keys = topK_keys))
    topK_feature = pd.concat(topK_feature, axis = 1)
    return topK_feature





def read_data():
    train_data = pd.read_csv("../../data/train/transaction_train_new.csv")
    test_data = pd.read_csv("../../data/test/test_transaction_round2.csv")
    trans_data = pd.concat([train_data, test_data])
    trans_data['trans_appro_geo_code'] = trans_data['geo_code'].apply(lambda x: x[0:2] if (x == x) else x)
    trans_data['trans_latitude'] = trans_data['geo_code'].apply(lambda x: geohash.decode_exactly(x)[0] if (x == x) else  x)
    trans_data['trans_longitude'] = trans_data['geo_code'].apply(lambda x: geohash.decode_exactly(x)[1] if (x == x) else  x)
    trans_data['trans_hour'] = trans_data['time'].apply(lambda x: int(x[0:2]))
    trans_data['trans_device_main_kind'] = trans_data['device2'].apply(lambda x: x.split(' ')[0] if (x == x) else x)

    #一些新的字段
    trans_data['hour_bin'] = trans_data['trans_hour'].apply(lambda x:hour_bin[x])
    trans_data['is_strange_mac1'] = trans_data['mac1'].apply(lambda x: 1 if x == 'a8dc52f65085212e' else 0)
    trans_data['is_strange_bal'] = trans_data['bal'].apply(lambda x: 1 if x == 100 else 0)
    return trans_data


def get_trans_new_feature():
    trans_data = read_data()

    #unstack feature
    trans_unstack_feature = get_trans_unstack_feature(trans_data) #3s
    #top_K feature
    #trans_topK_feature = get_trans_topK_feature(trans_data) #6s
    #topK_intersection_feature
    trans_intersection_topK_feature = get_trans_intersection_topK_feature() #10s
    trans_new_feature = pd.concat([trans_unstack_feature,  trans_intersection_topK_feature], axis=1)
    return trans_new_feature











