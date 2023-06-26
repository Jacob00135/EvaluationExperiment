import pandas as pd
import numpy as np

origin_all_datatype = []

origin_all = pd.read_csv('origin_all.csv')
for index in origin_all.dtypes.tolist():
    print(index)
    origin_all_datatype.append(index)
print(origin_all.dtypes)
origin_all_datatype = pd.DataFrame(origin_all_datatype)
origin_all_datatype.to_csv("origin_all_datatype.csv", index=0)

data = pd.read_csv('ADNI.csv')
origin_all_datatype = []
for index in data.dtypes.tolist():
    print(index)
    origin_all_datatype.append(index)

print(data.dtypes)
origin_all_datatype = pd.DataFrame(origin_all_datatype)
origin_all_datatype.to_csv("all_datatype.csv", index=0)