import pandas as pd

train = pd.read_csv('./ADNI.csv')
train.drop_duplicates(subset=['SavePath'], inplace=True)
train = train.loc[:, 'SavePath']
train.to_csv('./need_copy.txt', index=0)

