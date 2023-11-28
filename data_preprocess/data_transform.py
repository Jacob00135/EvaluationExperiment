import os
import pdb
import sys
import random
import numpy as np
import pandas as pd
from collections import Counter

now_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.realpath(os.path.join(now_path, '..'))
sys.path.append(root_path)

from config import warning_print


def clear_features(data):
    # 过滤缺失值过多的列：缺失值比例超过50%
    nan_filter = ["npiq_DEL", "npiq_HALL", "npiq_AGIT", "npiq_DEPD", "npiq_ANX", "npiq_ELAT", "npiq_APA", "npiq_DISN", "npiq_IRR", "npiq_MOT", "npiq_NITE", "npiq_APP", "his_CVHATT", "his_PSYCDIS", "his_ALCOHOL", "his_SMOKYRS", "his_PACKSPER"]
    data = data.drop(columns=nan_filter)

    # 过滤值单一的列：值单一比例超过85%
    value_filter = ["path", "SavePath", "PD", "FTD", "VD", "DLB", "PDD", "OTHER", "trailA", "trailB", "digitB", "digitBL", "digitF", "digitFL", "faq_STOVE", "his_NACCFAM", "his_CBSTROKE"]
    data = data.drop(columns=value_filter)

    # 其他处理
    drop_columns = ['NC', 'MCI', 'DE', 'AD', 'ADD', 'ALL']
    data['RID'] = data['RID'].values.astype('int32')
    data['COG'] = data['COG'].values.astype('int32')
    data = data.drop(columns=drop_columns)

    # 填充缺失值
    ls = ["gds", "lm_del", "mmse", "faq_BILLS", "faq_TAXES", "faq_SHOPPING", "faq_GAMES", "faq_MEALPREP", "faq_EVENTS", "faq_PAYATTN", "faq_REMDATES", "faq_TRAVEL", "his_DEPOTHR"]
    for c in ls:
        var = data[c].values
        notna_var = [v for v in var if not pd.isna(v)]
        fill_value = round(sum(notna_var) / len(notna_var), 0)
        data[c] = data[c].fillna(fill_value)

    return data


def split_dataset(data, save_path):
    num_sample = data.shape[0]
    train_sample = int(num_sample * 0.8)
    test_sample = num_sample - train_sample
    nan_benefit = pd.isna(data['benefit'].values)
    y = data['COG'].values
    info = np.zeros((4, 3), dtype='int')
    test_boolean = np.zeros(num_sample, dtype='bool')
    for c in range(info.shape[1]):
        category_boolean = y == c
        category_num = sum(category_boolean)
        train_category_num = int(category_num * 0.8)
        test_category_num = category_num - train_category_num
        nan_category_num = sum(nan_benefit & category_boolean)
        notnan_category_num = category_num - nan_category_num
        if notnan_category_num >= test_category_num:
            notnan_index = np.where((~nan_benefit) & category_boolean)[0]
            test_notnan_index = random.sample(notnan_index.tolist(), test_category_num)
            test_boolean[test_notnan_index] = True
        else:
            nan_index = np.where((nan_benefit & category_boolean))[0]
            test_nan_index = random.sample(nan_index.tolist(), len(nan_index) - train_category_num)
            test_boolean[test_nan_index] = True
            test_notnan_boolean = (~nan_benefit) & category_boolean
            test_boolean = test_boolean | test_notnan_boolean

    train_set = data[~test_boolean].copy()
    test_set = data[test_boolean].copy()
    test_set['benefit'] = test_set['benefit'].fillna(0)
    train_set.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    test_set.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    
    return train_set, test_set


if __name__ == '__main__':
    data_path = os.path.join(now_path, 'demo_character_process/data/ADNI_benifit.csv')
    data = pd.read_csv(data_path)
    data = clear_features(data)
    split_dataset(data, save_path=os.path.join(now_path, 'dataset'))
