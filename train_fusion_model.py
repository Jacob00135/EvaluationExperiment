import os
import pdb
import json
import torch
import numpy as np
import pandas as pd
from time import time as get_timestamp
from catboost import CatBoostRegressor
from config import root_path, mri_path
from models import _CNN_Bone, MLP


def mri_generator(filenames):
    for fn in filenames:
        path = os.path.join(mri_path, fn)
        mri = np.load(path).astype('float32')
        mri = np.expand_dims(np.expand_dims(mri, axis=0), axis=0)
        yield mri


def get_three_classes_prediction(scores, thresholds=(0.5, 1.5)):
    prediction = np.zeros(len(scores), 'int')
    for i in range(len(scores)):
        if scores[i] > thresholds[1]:
            prediction[i] = 2
        elif scores[i] >= thresholds[0]:
            prediction[i] = 1
    return prediction


def main():
    mri_model_name = 'MRI_model_20231124'  # 在此处修改MRI模型名称
    fusion_model_name = 'Fusion_model_20231124'  # 在此处修改Fusion模型名称

    # 加载MRI模型配置
    config_path = os.path.join(root_path, 'task_config.json')
    with open(config_path, 'r') as file:
        model_config = json.loads(file.read())
        file.close()

    # 加载MRI模型
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint_list_path = os.path.join(root_path, 'checkpoint_dir', mri_model_name)
    max_index = max([int(fn.rsplit('.', 1)[0].rsplit('_', 1)[1]) for fn in os.listdir(checkpoint_list_path)])
    backbone_path = os.path.join(checkpoint_list_path, 'backbone_{}.pth'.format(max_index))
    cog_path = os.path.join(checkpoint_list_path, 'COG_{}.pth'.format(max_index))
    backbone = _CNN_Bone(model_config['backbone']).to(device)
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    mlp = MLP(backbone.size, model_config['COG']).to(device)
    mlp.load_state_dict(torch.load(cog_path, map_location=device))
    print('已加载MRI模型')

    # 加载数据
    train_set_path = os.path.join(root_path, 'data_preprocess/dataset/train.csv')
    test_set_path = os.path.join(root_path, 'data_preprocess/dataset/test.csv')
    train_set = pd.read_csv(train_set_path)
    test_set = pd.read_csv(test_set_path)
    filenames = np.append(train_set['filename'].values, test_set['filename'].values)
    print('已加载数据')

    # 预测COG_Score
    backbone.train(False)
    mlp.train(False)
    prediction = np.zeros(filenames.shape[0], dtype='float32')
    with torch.no_grad():
        for i, mri in enumerate(mri_generator(filenames)):
            middle = backbone(torch.tensor(mri).to(device))
            prediction[i] = mlp(middle).data.cpu().squeeze().numpy()
    print('已预测COG_Score')

    # Fusion模型初始化
    train_set = train_set.drop(['RID', 'VISCODE', 'filename', 'benefit'], axis=1)
    test_set = test_set.drop(['RID', 'VISCODE', 'filename', 'benefit'], axis=1)
    train_set.insert(train_set.shape[1], 'COG_Score', prediction[:train_set.shape[0]])
    test_set.insert(test_set.shape[1], 'COG_Score', prediction[train_set.shape[0]:])
    x_train, y_train = train_set.drop(['COG'], axis=1).to_numpy(), train_set['COG'].values
    x_test, y_test = test_set.drop(['COG'], axis=1).to_numpy(), test_set['COG'].values
    fusion_model = CatBoostRegressor(iterations=1, learning_rate=0.05)
    checkpoint_dir = os.path.abspath(os.path.join(root_path, 'checkpoint_dir', fusion_model_name))
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    print('完成Fusion模型初始化')

    # 训练Fusion模型
    for epoch in range(100):
        # 训练、保存模型
        if epoch != 0:
            init_model = fusion_model
        else:
            init_model = None
        fusion_model.fit(x_train, y_train, init_model=init_model, verbose=False)
        fusion_model.save_model(os.path.join(checkpoint_dir, 'CatBoostRegressor_{}'.format(epoch)))

        # 验证模型
        pred_train = get_three_classes_prediction(fusion_model.predict(x_train))
        train_accuracy = sum(pred_train == y_train) / y_train.shape[0]
        pred_test = get_three_classes_prediction(fusion_model.predict(x_test))
        test_accuracy = sum(pred_test == y_test) / y_test.shape[0]
        print('Epoch {}: train_accuracy={:.4f} -- test_accuracy={:.4f}'.format(
            epoch + 1, train_accuracy, test_accuracy
        ))


if __name__ == '__main__':
    main()
