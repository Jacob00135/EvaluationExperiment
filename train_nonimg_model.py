import os
import numpy as np
import pandas as pd
from utils import read_json
from config import root_path
from catboost import CatBoostRegressor


def get_three_classes_prediction(scores, thresholds = (0.5, 1.5)):
    prediction = np.zeros(len(scores), 'int')
    for i in range(len(scores)):
        if scores[i] > thresholds[1]:
            prediction[i] = 2
        elif scores[i] >= thresholds[0]:
            prediction[i] = 1
    return prediction


def main():
    # 初始化
    model_name = 'nonimg_model_20231124'
    train_set = pd.read_csv(os.path.join(root_path, 'data_preprocess/dataset/train.csv'))
    test_set = pd.read_csv(os.path.join(root_path, 'data_preprocess/dataset/test.csv'))
    train_set = train_set.drop(['RID', 'VISCODE', 'filename', 'benefit'], axis=1)
    test_set = test_set.drop(['RID', 'VISCODE', 'filename', 'benefit'], axis=1)
    x_train, y_train = train_set.drop(['COG'], axis=1).to_numpy(), train_set['COG'].values
    x_test, y_test = test_set.drop(['COG'], axis=1).to_numpy(), test_set['COG'].values
    model = CatBoostRegressor(iterations=1, learning_rate=0.05)
    checkpoint_dir = os.path.abspath(os.path.join(root_path, 'checkpoint_dir', model_name))
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    for epoch in range(100):
        # 训练、保存模型
        if epoch != 0:
            init_model = model
        else:
            init_model = None
        model.fit(x_train, y_train, init_model=init_model, verbose=False)
        model.save_model(os.path.join(checkpoint_dir, 'CatBoostRegressor_{}'.format(epoch)))

        # 验证模型
        pred_train = get_three_classes_prediction(model.predict(x_train))
        train_accuracy = sum(pred_train == y_train) / y_train.shape[0]
        pred_test = get_three_classes_prediction(model.predict(x_test))
        test_accuracy = sum(pred_test == y_test) / y_test.shape[0]
        print('Epoch {}: train_accuracy={:.4f} -- test_accuracy={:.4f}'.format(
            epoch + 1, train_accuracy, test_accuracy
        ))


if __name__ == '__main__':
    main()
