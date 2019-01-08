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


label_test = pd.read_csv('../../data/submit_example.csv').drop('Tag', axis = 1)
label_train = pd.read_csv("../../data/train/tag_train_new.csv")
def feature_matrix_to_train_matrix(features):
    train = label_train.merge(features, how='left', left_on='UID',right_on = 'UID')
    train.fillna(0, inplace=True) 

    test = label_test.merge(features, how='left', left_on='UID',right_on = 'UID')
    test.fillna(0, inplace=True)


    X_train = train.drop(['Tag', 'UID'],axis=1)
    X_test = test.drop(['UID'],axis=1)
    y_train  = train['Tag']
    return X_train, X_test, y_train




def get_ope_amount_cnt(ope_data, primary_key = 'UID'):
    return ope_data.groupby([primary_key])[primary_key].count()



def ope_TopKOneHot_with_static(df, col_name, k = -1, primary_key = 'UID', fillna = -1, prefix = "topK_one_hot_", topK_keys = []):
    
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
    ope_UID_cnt = get_ope_amount_cnt(df)
    topK_one_hot[prefix + col_name +  '_THE_OTHER_ratio'] = topK_one_hot[prefix + col_name + '_unstack_THE_OTHER'] / ope_UID_cnt
    topK_one_hot[prefix + col_name + '_TopK_ratio'] = 1 - topK_one_hot[prefix + col_name + '_THE_OTHER_ratio']
    df[col_name] = tmp_col.copy()
    topK_feature = pd.concat([topK_one_hot, topK_static], axis = 1)
    return topK_feature


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


def get_ope_unstack_feature(ope_data):
    #unstack feature
    ope_unstack_col = [
        'success',
        'ope_hour',
        'os',
        'hour_bin',
        'is_strange_mac1',
        'is_strange_mac2',
        'is_strange_device_code3'
    ]
    ope_unstack_feature = []
    for col in ope_unstack_col:
        ope_unstack_feature.append(ope_data.groupby(['UID', col])[col].count().unstack(col).add_prefix('ope_' + col + '_unstack_'))
        ope_unstack_feature.append(ope_data.groupby(['UID', col])[col].count().groupby(['UID']).agg(
                                                        ['mean', 'std', 'max', 'min', 'median', np.ptp]).add_prefix('ope_' + col + '_static_'))
    ope_unstack_feature = pd.concat(ope_unstack_feature, axis = 1)
    return ope_unstack_feature




def get_ope_topK_feature(ope_data):
    #top_K feature
    ope_topK_col = [
        'mode',
        'device1',
        'device2',
        'ope_device_main_kind',

    ]

    ope_topK_k_value = {
        'mode': 30,
        'device1' :20,
        'device2': 35,
        'ope_device_main_kind': 25,
        'wifi': 10,
    }
    ope_topK_feature = []
    for col in ope_topK_col:
        ope_topK_feature.append(ope_TopKOneHot_with_static(ope_data, col_name = col, k = ope_topK_k_value[col],prefix = 'ope_'))
    ope_topK_feature = pd.concat(ope_topK_feature, axis = 1)
    return ope_topK_feature



#交集取测试集相交的前k个
def get_ope_intersection_topK_feature():
    

    #topK_intersection_feature
    ope_topK_intersection_col = [
        'version',
        'wifi',
        'ip1_sub',
    ]

    ope_topK_intersection_k_value = {
        'version': 15,
        'wifi': 10,
        'ip1_sub': 20,
    }
    train_data = pd.read_csv("../../data/train/operation_train_new.csv", 
                        dtype = {
                            'device1': str,
                            'geo_code':str
                        })
    test_data = pd.read_csv("../../data/test/operation_round1_new.csv",
                        dtype = {
                            'device1': str,
                            'geo_code':str
                        })
    ope_data = pd.concat([train_data, test_data])
    topK_names = ['merchant', 'code1', 'code2', 'acc_id1', 'device_code1', 'device_code2', 'device_code3',
                  'device1', 'device2', 'mac1', 'ip1', 'acc_id2', 'acc_id3', 'market_code', 'ip1_sub']
    topK_feature = []
    for col in ope_topK_intersection_col:
        k = ope_topK_intersection_k_value[col]
        target_list = train_data[col].value_counts().index
        topK_keys = [name for name in test_data[col].value_counts().index if name in target_list]
        if (len(topK_keys) > k):
            topK_keys = topK_keys[:k]
        topK_feature.append(ope_TopKOneHot_with_static(ope_data, col,primary_key = 'UID', prefix = "ope_intersection_" + col + '_', topK_keys = topK_keys))
    topK_feature = pd.concat(topK_feature, axis = 1)
    return topK_feature







def read_data():
    train_data = pd.read_csv("../../data/train/operation_train_new.csv")
    test_data = pd.read_csv("../../data/test/test_operation_round2.csv")
    ope_data = pd.concat([train_data, test_data])
    ope_data['ope_appro_geo_code'] = ope_data['geo_code'].apply(lambda x: x[0:2] if (x == x) else x)
    ope_data['ope_latitude'] = ope_data['geo_code'].apply(lambda x: geohash.decode_exactly(x)[0] if (x == x) else  x)
    ope_data['ope_longitude'] = ope_data['geo_code'].apply(lambda x: geohash.decode_exactly(x)[1] if (x == x) else  x)
    ope_data['ope_hour'] = ope_data['time'].apply(lambda x: int(x[0:2]))
    ope_data['ope_device_main_kind'] = ope_data['device2'].apply(lambda x: x.split(' ')[0] if (x == x) else x)
    ope_data['main_version'] = ope_data['version'].apply(lambda x: x.split('.')[0] if (x == x) else -1)
    #
    ope_data['os'] = ope_data['os'].astype(str)

    #一些新的字段
    ope_data['is_strange_mac1'] = ope_data['mac1'].apply(lambda x: 1 if x == 'a8dc52f65085212e' else 0)
    ope_data['is_strange_mac2'] = ope_data['mac2'].apply(lambda x: 1 if x == 'a8dc52f65085212e' else 0)
    ope_data['is_strange_device_code3'] = ope_data['device_code3'].apply(lambda x: 1 if x == '14c09cc8ce23d46c' else 0) 
    ope_data['hour_bin'] = ope_data['ope_hour'].apply(lambda x:hour_bin[x])
    return ope_data 





def get_ope_new_feature():
    ope_data = read_data()
    #unstack feature
    ope_unstack_feature = get_ope_unstack_feature(ope_data)
    #top_K feature
    #ope_topK_feature = get_ope_topK_feature(ope_data)
    #topK_intersection_feature
    ope_intersection_topK_feature = get_ope_intersection_topK_feature()
    ope_new_feature = pd.concat([ope_unstack_feature, ope_intersection_topK_feature], axis = 1)
    return ope_new_feature
















