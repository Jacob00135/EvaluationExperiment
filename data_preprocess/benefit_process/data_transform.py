import os
import sys
import pdb
import pandas as pd

"""
本文件需要完成的工作：
1. 添加benefit列
2. 筛选列变量
"""
now_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.realpath(os.path.join(now_path, '../..'))
data_path = os.path.realpath(os.path.join(now_path, 'benefit_data'))
benefit_filenames = [
    "our_cn_test.csv",
    "our_cn_train.csv",
    "our_Dementia_benifit_test.csv",
    "our_Dementia_benifit_train.csv",
    "our_Dementia_no_benifit_test.csv",
    "our_Dementia_no_benifit_train.csv",
    "our_MCI_benifit_test.csv",
    "our_MCI_benifit_train.csv",
    "our_MCI_no_benifit_test.csv",
    "our_MCI_no_benifit_train.csv"
]
sys.path.append(root_path)

from config import warning_print


if __name__ == '__main__':
    pass
