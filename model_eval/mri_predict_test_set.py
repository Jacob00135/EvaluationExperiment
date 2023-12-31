"""
使用保存的MRI模型对测试集进行预测，并保存预测结果
"""
import os
import pdb
import sys
import json
import torch
import numpy as np
import pandas as pd
from time import time as get_timestamp

now_path = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(now_path, '..')))
from config import root_path, mri_path
from models import _CNN_Bone, MLP


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


def main(model_name, test_set_path, result_save_path):
    # 根据下标整理模型
    checkpoint_path = os.path.join(root_path, 'checkpoint_dir', model_name)
    cp_dict = {}
    for filename in os.listdir(checkpoint_path):
        cp_type, index = filename[:filename.rfind('.')].split('_')
        if index in cp_dict:
            cp_dict[index][cp_type] = os.path.join(checkpoint_path, filename)
        else:
            cp_dict[index] = {cp_type: os.path.join(checkpoint_path, filename)}

    # 预测并保存
    filenames = pd.read_csv(test_set_path)['filename'].values
    paths = [os.path.join(mri_path, fn) for fn in filenames]
    result = np.zeros((len(cp_dict), len(paths)), 'float32')
    for i, (model_index, cp) in enumerate(cp_dict.items()):
        start_time = get_timestamp()
        mri_iter = mri_generator(paths)
        result[i] = MRIModel(cp['backbone'], cp['COG']).predict(mri_iter, len(paths))
        print('已预测：{} -- {:.0f}s'.format(model_index, get_timestamp() - start_time))
    np.save(result_save_path, result)


if __name__ == '__main__':
    main(
        model_name='MRI_model_20231124',  # 在此处修改模型名称
        test_set_path=os.path.join(root_path, 'data_preprocess/dataset/test.csv'),
        result_save_path=os.path.join(root_path, 'model_eval/eval_result/mri/scores.npy')
    )
