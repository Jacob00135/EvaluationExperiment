import os
import pdb
import sys
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

    if nan_benefit.sum() > train_sample:
        nan_index = np.where(nan_benefit)[0]
        np.random.shuffle(nan_index)
        train_index = nan_index[:train_sample]
        train_boolean = np.zeros(num_sample, dtype='bool')
        train_boolean[train_index] = True
        train_set = data[train_boolean]
        test_set = data[~train_boolean]
        test_set['benefit'] = test_set['benefit'].fillna(0)
    else:
        notna_benefit = ~nan_benefit
        notna_index = np.where(notna_benefit)[0]
        np.random.shuffle(notna_index)
        test_index = notna_index[:test_sample]
        test_boolean = np.zeros(num_sample, dtype='bool')
        test_boolean[test_index] = True
        train_set = data[~test_boolean]
        test_set = data[test_boolean]

    train_set.to_csv(os.path.join(save_path, 'train.csv'))
    test_set.to_csv(os.path.join(save_path, 'test.csv'))
    
    return train_set, test_set


if __name__ == '__main__':
    data_path = os.path.join(now_path, 'demo_character_process/data/ADNI_benifit.csv')
    data = pd.read_csv(data_path)
    data = clear_features(data)
    split_dataset(data, save_path=os.path.join(now_path, 'dataset'))
