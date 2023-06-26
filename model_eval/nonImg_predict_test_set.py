"""
使用保存的nonImg、Fusion模型对测试集进行预测，并保存预测结果
"""
import os
import sys
import pdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

now_path = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(now_path, '..')))
from config import root_path


def main(test_set_path, checkpoint_dir_path, result_save_path):
    # 载入数据
    test_set = pd.read_csv(test_set_path)
    x_test = test_set.drop(['COG'], axis=1).to_numpy()

    # 遍历模型预测
    filenames = sorted(os.listdir(checkpoint_dir_path), key=lambda v: int(v.rsplit('_', 1)[1]))
    result = np.zeros((len(filenames), test_set.shape[0]), 'float32')
    for i, fn in enumerate(filenames):
        model = CatBoostRegressor()
        model.load_model(os.path.join(checkpoint_dir_path, fn))
        result[i] = model.predict(x_test)
    np.save(result_save_path, result)


if __name__ == '__main__':
    main(
        test_set_path=os.path.join(root_path, 'lookupcsv/CrossValid/no_cross/fusion_test.csv'),
        checkpoint_dir_path=os.path.join(root_path, 'checkpoint_dir/Fusion_v1'),
        result_save_path=os.path.join(root_path, 'model_eval/eval_result/Fusion/scores.npy')
    )
