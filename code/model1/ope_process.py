#-*- encoding:utf-8 -*-
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



#每个用户的操作总数（常用）
def get_ope_amount_cnt(ope_data, primary_key = 'UID'):
    return ope_data.groupby([primary_key])[primary_key].count()




def get_ope_popular_degree(ope_data, primary_key = 'UID'):
    pop_degree_cols = [
        'version', 'device1', 'device2', 'device_code1', 'device_code2', 'device_code3',
        'mac1', 'ip1', 'ip2', 'wifi', 'ip1_sub', 'ip2_sub'
    ]
    for col in pop_degree_cols:
        col_popular_degree = ope_data.groupby([col])[primary_key].agg({'unique_count': lambda x: len(pd.unique(x))})
        tmp = col_popular_degree.reset_index()
        tmp.set_index(col, inplace=True)
        data_dict = tmp.to_dict()
        ope_data['ope_' + col + '_pop_degree'] = ope_data[col].apply(
                                                        lambda x: data_dict['unique_count'][x] if (x == x) else x)



def read_data():
    pop_col_names = ['day', 'mode', 'ope_hour', 'version', 'device_code1', 'device_code2', 'device_code3',
                     'mac1', 'ip1', 'ip2', 'mac2', 'wifi', 'ope_appro_geo_code', 'ip1_sub', 'ip2_sub']
    train_data = pd.read_csv("../../data/train/operation_train_new.csv", 
                        dtype = {
                            'device1': str,
                            'geo_code':str
                        })
    train_data['ope_appro_geo_code'] = train_data['geo_code'].apply(lambda x: x[0:2] if (x == x) else x)
    train_data['ope_hour'] = train_data['time'].apply(lambda x: int(x[0:2]))


    test_data = pd.read_csv("../../data/test/test_operation_round2.csv",
                        dtype = {
                            'device1': str,
                            'geo_code':str
                        })
    test_data['ope_appro_geo_code'] = test_data['geo_code'].apply(lambda x: x[0:2] if (x == x) else x)

    test_data['ope_hour'] = test_data['time'].apply(lambda x: int(x[0:2]))
    for col in pop_col_names:
        get_ope_popular_degree(ope_data= train_data)
        get_ope_popular_degree(ope_data=test_data)

    #sampling完全没用 结果大幅下降
    #test_data = test_data.sample(n=len(train_data), random_state=0)

    ope_data = pd.concat([train_data, test_data])
    ope_data['ope_latitude'] = ope_data['geo_code'].apply(lambda x: geohash.decode_exactly(x)[0] if (x == x) else  x)
    ope_data['ope_longitude'] = ope_data['geo_code'].apply(lambda x: geohash.decode_exactly(x)[1] if (x == x) else  x)
    #get_ip_pop_degree(ope_data)
    
    
    return ope_data


def get_ope_pop_degree_feature(ope_data):
    pop_degree_cols = [
        'version', 'device1', 'device2', 'device_code1', 'device_code2', 'device_code3',
        'mac1', 'ip1', 'wifi', 'ip1_sub'
    ]
    pop_col_names = ['ope_' + col + '_pop_degree' for col in pop_degree_cols]
    pop_degree_feature= []
    for col in pop_col_names:
        pop_degree_feature.append(ope_data.groupby(['UID'])[col].agg(
                                        {'mode': lambda x: stats.mode(x)[0][0]}).add_prefix(col + '_static_'))
        pop_degree_feature.append(ope_data.groupby(['UID'])[col].agg(
                                        ['max', 'mean', 'std', 'median']).add_prefix(col + '_static_'))
    pop_degree_feature = pd.concat(pop_degree_feature, axis = 1)
    return pop_degree_feature








def get_ope_latitude_longitude_feature(ope_data):
    ope_geo_latitude_static = ope_data.groupby(['UID'])['ope_latitude'].agg(['max', 'min', 'std']).add_prefix('ope_geo_latitude_static_')
    ope_geo_longitude_static = ope_data.groupby(['UID'])['ope_longitude'].agg(['max', 'min', 'std']).add_prefix('ope_geo_longitude_static_')
    ope_geo_latitude_static['ope_geo_latitude_min_max_diff'] = ope_geo_latitude_static['ope_geo_latitude_static_max'] - ope_geo_latitude_static['ope_geo_latitude_static_min']
    ope_geo_longitude_static['ope_geo_longitude_min_max_diff'] = ope_geo_longitude_static['ope_geo_longitude_static_max'] - ope_geo_longitude_static['ope_geo_longitude_static_min']

    ope_geo_latitude_daily_diff = ope_data.groupby(['UID', 'day'])['ope_latitude'].agg(
                                    {'': lambda x: np.max(x) - np.min(x)}).groupby('UID').agg(
                                    ['max', 'std', 'mean']).add_prefix('ope_geo_latitude_daily_diff_')['ope_geo_latitude_daily_diff_']
    ope_geo_longitude_daily_diff = ope_data.groupby(['UID', 'day'])['ope_longitude'].agg(
                                    {'': lambda x: np.max(x) - np.min(x)}).groupby('UID').agg(
                                    ['max', 'std', 'mean']).add_prefix('ope_geo_longitude_daily_diff_')['ope_geo_longitude_daily_diff_']

    ope_geo_latitude_longitude_feature = pd.concat([ope_geo_latitude_static, ope_geo_longitude_static, 
                                                   ope_geo_latitude_daily_diff, ope_geo_longitude_daily_diff,
                                                   ], axis = 1)

    return ope_geo_latitude_longitude_feature





