import os
import pdb
import numpy as np
import pandas as pd
from compute_performance import compute_ADAS_benefit

root_path = os.path.realpath(os.path.dirname(__file__))


def match_drug_info():
    # 载入数据
    test_path = os.path.join(root_path, '../lookupcsv/CrossValid/no_cross/test_source.csv')
    test_set = pd.read_csv(test_path)
    drug_data_path = os.path.join(root_path, 'data/BACKMEDS.csv')
    drug_data2_path = os.path.join(root_path, 'data/RECCMEDS.csv')
    drug_data = pd.read_csv(drug_data_path)
    drug_data2 = pd.read_csv(drug_data2_path, low_memory=False)
    test_set = test_set[['RID']].drop_duplicates(keep='first')
    drug_data = drug_data[['RID', 'KEYMED']]
    drug_data2 = drug_data2[['RID', 'CMMED']]
    test_set.index = range(test_set.shape[0])

    # 将药物列变量值转换成字符串
    drug_map = [
        'Other',
        'Aricept',
        'Cognex',
        'Exelon',
        'Namenda',
        'Razadyne',
        'Anti-depressant medication',
        'Other behavioral medication',
    ]
    drug = drug_data['KEYMED'].values
    for i, v in enumerate(drug):
        if pd.isna(v):
            drug[i] = np.nan
            continue
        is_number = True
        try:
            v = int(v)
        except Exception:
            is_number = False
        if is_number:
            drug[i] = drug_map[v]
            continue
        if ':' in v:
            number_list = v.split(':')
        elif '|' in v:
            number_list = v.split('|')
        number_list = sorted([int(num) for num in number_list])
        if number_list[0] == 0:
            number_list.pop(0)
            number_list.append(0)
        drug[i] = ', '.join([drug_map[num] for num in number_list])
    
    # 处理-4空缺值
    drug = drug_data2['CMMED'].values
    for i, v in enumerate(drug):
        if pd.isna(v) or v == '-4':
            drug[i] = 'Other'

    # 匹配
    data = pd.merge(test_set, drug_data, how='left', on=['RID'])
    data = pd.merge(data, drug_data2, how='left', on=['RID'])
    drug = np.zeros(data.shape[0], dtype='object')
    k1 = data['KEYMED'].values
    k2 = data['CMMED'].values
    for i in range(drug.shape[0]):
        v1 = k1[i]
        v2 = k2[i]
        if pd.notna(v1) and pd.notna(v2):
            drug[i] = ', '.join([v1, v2])
        elif pd.notna(v1):
            drug[i] = v1
        elif pd.notna(v2):
            drug[i] = v2
        else:
            drug[i] = np.nan
    data = pd.DataFrame({'RID': data['RID'].values, 'DRUG': drug})

    # 整理病人所用药物
    data = data.sort_values(by='RID')
    data.index = range(data.shape[0])
    delete_index = []
    i = 0
    while i < data.shape[0]:
        # 搜索同一个患者的药物
        rid = data.loc[i, 'RID']
        drug_list = [data.loc[i, 'DRUG']]
        j = i + 1
        while j < data.shape[0] and data.loc[j, 'RID'] == rid:
            drug_list.append(data.loc[j, 'DRUG'])
            delete_index.append(j)
            j = j + 1

        # 整理值
        drug_set = set()
        for multi_drug in drug_list:
            for v in multi_drug.split(', '):
                drug_set.add(v)
        drug_list = sorted(drug_set)
        if 'Other' in drug_list:
            drug_list.remove('Other')
            drug_list.append('Other')
        data.loc[i, 'DRUG'] = ', '.join(drug_list)
        i = j
    data.index = range(data.shape[0])
    data = data.drop(delete_index)
    data.sort_values(by='RID')
    data.index = range(data.shape[0])

    # 导出
    data.to_csv(os.path.join(root_path, 'data/PatientDrugStatistics.csv'), index=False)


if __name__ == '__main__':
    match_drug_info()
