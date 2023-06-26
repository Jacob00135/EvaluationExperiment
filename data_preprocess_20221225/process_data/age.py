import pandas as pd
import numpy as np
import copy
import csv

nature_img = pd.read_csv('../raw_data/nature_imginfo.csv')
merge = pd.read_csv('../raw_data/ADNIMERGE_without_bad_value.csv')
PTDEMOG = pd.read_csv('../raw_data/PTDEMOG.csv')

data = pd.merge(nature_img, merge, on=['RID', 'VISCODE'], how='left') #出现sc, scmri的记录没有匹配数据的情况,merge表中的年龄为第一次随访年龄

# def calculate_age(data):

df = copy.deepcopy(data)
ridlist = {}

#人口统计学
for index, row in data.iterrows():
    if (row['RID'] not in ridlist.keys()) and (row['AGE'] == row['AGE']): #找出每个RID受试者的一次有效AGE记录
        # print("row['AGE']= ", row['AGE'])
        ridlist[row['RID']] = index
print("len(ridlist)= ", len(ridlist))
for index, row in data.iterrows():
    if row['RID'] in ridlist.keys() and (row['AGE'] != row['AGE']): #以AGE作为基本人口信息的标志，若为空，则先填补
        inx = ridlist[row['RID']]
        # print(data.loc[index, 'AGE'])

        if (row['VISCODE'] == 'sc') | (row['VISCODE'] == 'scmri'):
            df.loc[index, 'AGE'] = data.loc[inx, 'AGE']
            print("======")
        df.loc[index, 'PTGENDER'] = data.loc[inx, 'PTGENDER']
        df.loc[index, 'PTEDUCAT'] = data.loc[inx, 'PTEDUCAT']
        df.loc[index, 'PTETHCAT'] = data.loc[inx, 'PTETHCAT']
        df.loc[index, 'PTRACCAT'] = data.loc[inx, 'PTRACCAT']
        df.loc[index, 'APOE4'] = data.loc[inx, 'APOE4']
        df.loc[index, 'DX'] = data.loc[inx, 'DX']
        df.loc[index, 'FLDSTRENG'] = data.loc[inx, 'FLDSTRENG']
        df.loc[index, 'MOCA'] = data.loc[inx, 'MOCA']
#-----上面解决初步用merge表补全部分数据的人口统计学信息，还有559条缺少

PTDEMOG = pd.read_csv('../raw_data/PTDEMOG.csv')
PTDEMOG_ird_viscode = PTDEMOG.loc[:, ['RID', 'PTDOBYY', 'USERDATE']]
df = pd.merge(df, PTDEMOG_ird_viscode, on=['RID'], how='left')
data = copy.deepcopy(df)
i = 0
j = 0
for index, row in df.iterrows():
    if (row['EXAMDATE'] == row['EXAMDATE']) and (row['PTDOBYY'] == row['PTDOBYY']):
        data.loc[index, 'AGE'] = int(row['EXAMDATE'][:4]) - int(row['PTDOBYY'])
    elif (row['EXAMDATE'] != row['EXAMDATE']) and (row['PTDOBYY'] == row['PTDOBYY']):
        data.loc[index, 'AGE'] = int(row['USERDATE'][:4]) - int(row['PTDOBYY'])
    else:
        i += 1
        if (row['VISCODE'] != 'sc') | (row['VISCODE'] != 'scmri'):
            data = data.drop(index)
            j += 1
print("i= ", i)
print("j= ", j)

data = data.drop(['PTDOBYY'], axis=1)
data.drop_duplicates(subset=['RID','VISCODE'],keep='first',inplace=True)

PTDEMOG = pd.read_csv('../raw_data/PTDEMOG.csv')
PTDEMOG = PTDEMOG.rename(columns={'VISCODE': 'VISCODE1', 'VISCODE2':'VISCODE'})
df = pd.merge(data, PTDEMOG, on=['RID', 'VISCODE'], how='left')
data = copy.deepcopy(df)
data.to_csv('age.csv', index=0)
for index, row in df.iterrows():
    if row['PTDOBYY'] != row['PTDOBYY']:
        data.loc[index, 'PTDOBYY'] = 0
    if row['PTEDUCAT_y'] != row['PTEDUCAT_y']:
        data.loc[index, 'PTEDUCAT_x'] = 0
    else:
        data.loc[index, 'PTEDUCAT_x'] = int(row['PTEDUCAT_y'])
    if row['PTETHCAT_y'] !=  row['PTETHCAT_y']:
        data.loc[index, 'PTETHCAT_x'] = 0
    else:
        data.loc[index, 'PTETHCAT_x'] = int(row['PTETHCAT_y'])
    race = ''

    if str(row['PTRACCAT_y']) == '1.0':
        race = 'ind'
    elif str(row['PTRACCAT_y']) == '2.0':
        race = 'ans'
    elif str(row['PTRACCAT_y']) == '3.0':
        race = 'haw'
    elif str(row['PTRACCAT_y']) == '4.0':
        race = 'blk'
    elif str(row['PTRACCAT_y']) == '5.0':
        race = 'whi'
    elif str(row['PTRACCAT_y']) == '6.0':
        race = 'mix'
    data.loc[index, 'PTRACCAT_x'] = race
    data.loc[index, 'PTGENDER_x'] = 'male' if str(row['PTGENDER_y']) == '1.0' else 'female'



data = data.drop(['PTEDUCAT_y', 'PTETHCAT_y', 'PTRACCAT_y', 'PTGENDER_y'], axis=1)
data = data.rename(columns={ 'PTEDUCAT_x':'PTEDUCAT', 'PTETHCAT_x':'PTETHCAT', 'PTRACCAT_x':'PTRACCAT', 'PTGENDER_x':'PTGENDER'})











# def readcsv(csv_file):
#     csvfile = open(csv_file, 'r')
#     return csv.DictReader(csvfile)
#
# PTDEMOG = readcsv('../raw_data/PTDEMOG.csv')
# ridlist = []
# for row in PTDEMOG:
#     if row['RID'] not in ridlist :
#
#         if row['PTDOBYY'] == '':
#             row['PTDOBYY'] = 0
#         if row['PTEDUCAT'] == '':
#             row['PTEDUCAT'] = 0
#         if row['PTETHCAT'] == '':
#             row['PTETHCAT'] = 0
#         race = ''
#         if row['PTRACCAT'] == '1':
#             race = 'ind'
#         elif row['PTRACCAT'] == '2':
#             race = 'ans'
#         elif row['PTRACCAT'] == '3':
#             race = 'haw'
#         elif row['PTRACCAT'] == '4':
#             race = 'blk'
#         elif row['PTRACCAT'] == '5':
#             race = 'whi'
#         elif row['PTRACCAT'] == '6':
#             race = 'mix'
#
#         if data[(data['RID'] == int(row['RID'])) & ((data['VISCODE'] == 'sc') | (data['VISCODE'] == 'scmri'))].shape[0] != 0:
#             index = data[(data['RID'] == int(row['RID'])) & ((data['VISCODE'] == 'sc') | (data['VISCODE'] == 'scmri'))].index.tolist()[0]
#
#             # data.loc[index, 'AGE'] = int(row['USERDATE'][:4]) - int(row['PTDOBYY'])
#             data.loc[index, 'PTGENDER'] = 'male' if row['PTGENDER'] == '1' else 'female'
#             data.loc[index, 'PTEDUCAT'] = int(row['PTEDUCAT'])
#             data.loc[index, 'PTETHCAT'] = int(row['PTETHCAT'])
#             data.loc[index, 'PTRACCAT'] = race
#
#             ridlist.append(row['RID'])
#             print("进来了！！！")

#
# data = copy.deepcopy(df)
# PTDEMOG = pd.read_csv('../raw_data/PTDEMOG.csv')
# # PTDEMOG_ird_viscode = PTDEMOG.loc[:, ['RID', 'PTDOBYY']]
# for index, row in df.iterrows():
#     if ((row['VISCODE'] != 'sc') & (row['VISCODE'] != 'scmri')):
#         idx = PTDEMOG[PTDEMOG['RID'] == int(row['RID'])].index.tolist()[0]
#         data.loc[index, 'AGE'] = int(df.loc[index, 'EXAMDATE'][:4]) - int(PTDEMOG.loc[idx, 'PTDOBYY'])
#
#

# data = data.loc[:, ['RID', 'VISCODE', 'AGE', 'USERDATE', 'PTDOBYY']]

data.to_csv('age.csv', index=0)
