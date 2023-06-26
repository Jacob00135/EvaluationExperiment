"""
使用保存的MRI模型对测试集进行预测，并保存预测结果
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from time import time as get_timestamp
from models import _CNN_Bone, MLP

now_path = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(now_path, '..')))
from config import root_path


class MRIModel(object):
    model_config = None

    def __init__(self, backbone_path, cog_path):
        self.load_config()
        
        self.backbone_path = backbone_path
        self.cog_path = cog_path
        self.backbone = None
        self.mlp = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    @classmethod
    def load_config(cls):
        if cls.model_config is not None:
            return None
        with open(os.path.join(root_path, 'task_config.json'), 'r') as file:
            cls.model_config = json.loads(file.read())
            file.close()

    def load_model(self):
        self.backbone = _CNN_Bone(self.model_config['backbone']).to(self.device)
        self.backbone.load_state_dict(torch.load(self.backbone_path, map_location='cuda:0'))
        self.mlp = MLP(self.backbone.size, self.model_config['COG']).to(self.device)
        self.mlp.load_state_dict(torch.load(self.cog_path, map_location='cuda:0'))

    def predict(self, mri_iter, num_mri):
        self.backbone.train(False)
        self.mlp.train(False)
        result = np.zeros(num_mri, 'float32')
        with torch.no_grad():
            for i, mri in enumerate(mri_iter):
                middle = self.backbone(torch.tensor(mri).to(self.device))
                result[i] = self.mlp(middle).data.cpu().squeeze().numpy()
        return result


def mri_generator(paths):
    for path in paths:
        mri = np.load(path).astype('float32')
        mri = np.expand_dims(np.expand_dims(mri, axis=0), axis=0)
        yield mri


def main(checkpoint_path, test_set_path, mri_path, result_save_path):
    # 根据下标整理模型
    cp_dict = {}
    for filename in os.listdir(checkpoint_path):
        cp_type, index = filename[:filename.rfind('.')].split('_')
        if index in cp_dict:
            cp_dict[index][cp_type] = os.path.join(checkpoint_path, filename)
        else:
            cp_dict[index] = {cp_type: os.path.join(checkpoint_path, filename)}

    # 载入MRI数据（生成器）
    test_set = pd.read_csv(test_set_path)
    mri_path_generator = map(lambda fn: os.path.join(mri_path, fn), test_set['filename'].values)
    mri_iter = mri_generator(mri_path_generator)

    # 预测并保存
    result = np.zeros((len(cp_dict), len(test_set)), 'float32')
    for i, (model_index, cp) in enumerate(cp_dict.items()):
        start_time = get_timestamp()
        save_path = os.path.join(result_save_path, '{}.npy'.format(model_index))
        if os.path.exists(save_path):
            continue
        result[i] = MRIModel(cp['backbone'], cp['COG']).predict(mri_iter, len(test_set))
        print('已预测：{} -- {:.0f}s'.format(model_index, get_timestamp() - start_time))
    np.save(result_save_path, result)


if __name__ == '__main__':
    main(
        checkpoint_path=os.path.join(root_path, 'checkpoint_dir/MRI_only_v3'),
        test_set_path=os.path.join(root_path, 'lookupcsv/CrossValid/no_cross/test.csv'),
        mri_path='/home/lxs/ADNI/npy/',  # 172.29.23.249:/home/lxs/ADNI/npy/
        result_save_path=os.path.join(root_path, 'model_eval/eval_result/mri/scores.npy')
    )
