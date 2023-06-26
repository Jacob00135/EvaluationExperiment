import os
import pdb
import json
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from config import root_path, feature_columns, required_columns


def read_task_config():
    with open(os.path.join(root_path, 'nonImg_task_config.json'), 'r') as file:
        task_config = json.loads(file.read())
        file.close()
    return task_config


def preprocess(dataset_path):
    dataset = pd.read_csv(dataset_path)[required_columns]
    if 'gender' in dataset:
        data['gender'] = data['gender'].map({'male': 0, 'female': 1})
    pdb.set_trace()
    features = dataset[feature_columns]
    features = IterativeImputer(max_iter=1000).fit(features).transform(features)
    for c in range(features.shape[1]):
        var = features[:, c]
        std = var.std()
        if std:
            features[:, c] = (var - var.mean()) / std
    dataset = pd.concat((
        dataset[['RID', 'VISCODE', 'COG']],
        pd.DataFrame(features, columns=list(set(feature_columns) & set(dataset.columns)))
    ), axis=1)
    return dataset


def main(data_path):
    preprocess(os.path.join(data_path, 'test_source.csv')).to_csv(
        os.path.join(data_path, 'risk_factor_test.csv'),
        index=False
    )


if __name__ == '__main__':
    main(
        data_path=os.path.join(root_path, 'lookupcsv/CrossValid/no_cross')
    )
