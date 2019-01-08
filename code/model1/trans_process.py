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


def tpr_weight_funtion(preds,dtrain):
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
    #return "res", TR1
    return "res", 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3



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


def get_trans_amount_cnt(trans_data, primary_key = 'UID'):
    return trans_data.groupby([primary_key])[primary_key].count()



def get_trans_popular_degree(trans_data, primary_key = 'UID'):
    pop_degree_cols = [
        'acc_id1','acc_id2','acc_id3',
        'device_code1','device_code2','device_code3',
        'device1','device2', 'mac1',
        'ip1', 'ip1_sub'
    ]
    for col in pop_degree_cols:
        col_popular_degree = trans_data.groupby([col])[primary_key].agg({'unique_count': lambda x: len(pd.unique(x))})
        tmp = col_popular_degree.reset_index()
        tmp.set_index(col, inplace=True)
        data_dict = tmp.to_dict()
        trans_data['trans_' + col + '_pop_degree'] = trans_data[col].apply(
                                                        lambda x: data_dict['unique_count'][x] if (x == x) else x)



def get_trans_pop_degree_feature(trans_data):
    pop_degree_cols = [
        'acc_id1','acc_id2','acc_id3',
        'device_code1','device_code2','device_code3',
        'device1','device2', 'mac1',
        'ip1', 'ip1_sub'
    ]
    pop_col_names = ['trans_' + col + '_pop_degree' for col in pop_degree_cols]
    pop_degree_feature= []
    for col in pop_col_names:
        pop_degree_feature.append(trans_data.groupby(['UID'])[col].agg(
                                        {'mode': lambda x: stats.mode(x)[0][0]}).add_prefix(col + '_static_'))
        pop_degree_feature.append(trans_data.groupby(['UID'])[col].agg(
                                        ['max', 'mean', 'std', 'median']).add_prefix(col + '_static_'))
    pop_degree_feature = pd.concat(pop_degree_feature, axis = 1)
    return pop_degree_feature



def read_data():
    pop_col_names = ['day', 'merchant', 'code1', 'code2', 'acc_id1', 'device_code1', 'device_code2', 'device_code3',
                     'mac1', 'ip1', 'acc_id2', 'acc_id3', 'market_code', 'ip1_sub']
    
    train_data = pd.read_csv("../../data/train/transaction_train_new.csv", 
                        dtype = {
                            'device1': str,
                            'geo_code':str
                        })
    train_data['trans_appro_geo_code'] = train_data['geo_code'].apply(lambda x: x[0:2] if (x == x) else x)
    #transfrom_version_info(train_data)
    test_data = pd.read_csv("../../data/test/test_transaction_round2.csv",
                        dtype = {
                            'device1': str,
                            'geo_code':str
                        })
    for col in pop_col_names:
        get_trans_popular_degree(trans_data=train_data)
        get_trans_popular_degree(trans_data=test_data)
    #transfrom_version_info(test_data)
    train_data['trans_appro_geo_code'] = train_data['geo_code'].apply(lambda x: x[0:2] if (x == x) else x)
    
    trans_data = pd.concat([train_data, test_data])
    
    trans_data['trans_latitude'] = trans_data['geo_code'].apply(lambda x: geohash.decode_exactly(x)[0] if (x == x) else  x)
    trans_data['trans_longitude'] = trans_data['geo_code'].apply(lambda x: geohash.decode_exactly(x)[1] if (x == x) else  x)
    trans_data['trans_hour'] = trans_data['time'].apply(lambda x: int(x[0:2]))
    #get_ip_pop_degree(trans_data)
    #get_trans_popular_degree(trans_data =trans_data, col='ip1')
    #get_trans_popular_degree(trans_data =trans_data, col='ip2')
    return trans_data



def get_trans_latitude_longitude_feature(trans_data):
    trans_geo_latitude_static = trans_data.groupby(['UID'])['trans_latitude'].agg(['max', 'min', 'std']).add_prefix('trans_geo_latitude_static_')
    trans_geo_longitude_static = trans_data.groupby(['UID'])['trans_longitude'].agg(['max', 'min', 'std']).add_prefix('trans_geo_longitude_static_')
    trans_geo_latitude_static['trans_geo_latitude_min_max_diff'] = trans_geo_latitude_static['trans_geo_latitude_static_max'] - trans_geo_latitude_static['trans_geo_latitude_static_min']
    trans_geo_longitude_static['trans_geo_longitude_min_max_diff'] = trans_geo_longitude_static['trans_geo_longitude_static_max'] - trans_geo_longitude_static['trans_geo_longitude_static_min']

    trans_geo_latitude_daily_diff = trans_data.groupby(['UID', 'day'])['trans_latitude'].agg(
                                    {'': lambda x: np.max(x) - np.min(x)}).groupby('UID').agg(
                                    ['max', 'std', 'mean']).add_prefix('trans_geo_latitude_daily_diff_')['trans_geo_latitude_daily_diff_']
    trans_geo_longitude_daily_diff = trans_data.groupby(['UID', 'day'])['trans_longitude'].agg(
                                    {'': lambda x: np.max(x) - np.min(x)}).groupby('UID').agg(
                                    ['max', 'std', 'mean']).add_prefix('trans_geo_longitude_daily_diff_')['trans_geo_longitude_daily_diff_']

    trans_geo_latitude_longitude_feature = pd.concat([trans_geo_latitude_static, trans_geo_longitude_static, 
                                                   trans_geo_latitude_daily_diff, trans_geo_longitude_daily_diff,
                                                   ], axis = 1)

    return trans_geo_latitude_longitude_feature
