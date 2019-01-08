
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np


file_names = [
    './model1/result/model1_res.csv',
    './model2/result/78508.csv'
]
res_col_name = ['model1_res_res', 'model2_res_res']
res_col_name.insert(0, 'UID')



res = []
UID = []

for file in file_names:
    tmp_res = pd.read_csv(file)
    UID = tmp_res['UID']
    res.append(tmp_res['Tag'])
res.insert(0, UID)
res = pd.concat(res ,axis = 1)
res.columns = res_col_name



ratio = res['model2_res_res'].mean() / res['model1_res_res'].mean()
res['model2_res_res'] =  res['model2_res_res'].apply(lambda x: x / ratio)



res['Tag'] = res[['model1_res_res','model2_res_res']].apply(lambda x: (x[0] ** 0.4) * (x[1] ** 0.5),axis=1)



res.to_csv('../result/final_res.csv', index = False, columns=['UID', 'Tag'])
print('ensemble,done')

