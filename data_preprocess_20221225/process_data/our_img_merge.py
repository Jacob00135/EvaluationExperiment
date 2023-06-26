import pandas as pd
import numpy as np
import copy
import csv

nature_img = pd.read_csv('../raw_data/nature_imginfo.csv')
merge = pd.read_csv('../raw_data/ADNIMERGE_without_bad_value.csv')
PTDEMOG = pd.read_csv('../raw_data/PTDEMOG.csv')

data = pd.merge(nature_img, merge, on=['RID', 'VISCODE'], how='left') #出现sc, scmri的记录没有匹配数据的情况,merge表中的年龄为第一次随访年龄

# def calculate_age(data):
columnslist=['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER','cdr', 'cdrSum', 'Tesla'
                                              , 'adas_q1', 'adas_q2', 'adas_q3', 'adas_q4', 'adas_q5', 'adas_q6', 'adas_q7', 'adas_q8', 'adas_q9',
                 'adas_q10', 'adas_q11', 'adas_q12', 'adas_q14', 'adas_total11', 'adas_totalmod',
                                              'trailA', 'trailB',
                                              'lm_imm', 'lm_del', 'boston', 'animal', 'vege',
                                              'digitB', 'digitBL', 'digitF', 'digitFL',
                                            'moca', 'moca_execu', 'moca_visuo', 'moca_name', 'moca_atten', 'moca_senrep', 'moca_verba',
                     'moca_abstr', 'moca_delrec', 'moca_orient',
'npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD',
                 'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN',
                 'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP',
"his_CVAFIB","his_CVANGIO","his_CVBYPASS","his_CVPACE", "his_CVCHF","his_CVOTHR", "his_CBTIA","his_SEIZURES",
"his_TBI", "his_HYPERCHO", "his_DIABETES", "his_B12DEF", "his_THYROID", "his_INCONTU", "his_INCONTF", "his_DEP2YRS",
"his_TOBAC100", "his_ABUSOTHR", "COG_score", "ADD_score"
                                              ]
new_columns = pd.DataFrame(columns=columnslist)
new_columns = new_columns.reindex(columns=columnslist ,fill_value=0)
data = pd.concat([data, new_columns], sort=False)

#修正字段列名
def correct_columnname(data):
    df = copy.deepcopy(data)
    for column in df.columns.tolist():
        if column[-2 :] == '_y':
            data = data.drop([column], axis=1)
            # print(column)
        if column[-2:] == '_x':
            new_column = column[:-2]
            data = data.rename(columns={column: new_column})
    return data

df = copy.deepcopy(data)
ridlist = {}

#人口统计学
for index, row in data.iterrows():
    if (row['RID'] not in ridlist.keys()) and (row['AGE'] == row['AGE']): #找出每个RID受试者的一次有效AGE记录
        # print("row['AGE']= ", row['AGE'])
        ridlist[row['RID']] = index
# print("len(ridlist)= ", len(ridlist))
for index, row in data.iterrows():
    if row['RID'] in ridlist.keys() and (row['AGE'] != row['AGE']): #以AGE作为基本人口信息的标志，若为空，则先填补
        inx = ridlist[row['RID']]
        # print(data.loc[index, 'AGE'])

        if (row['VISCODE'] == 'sc') | (row['VISCODE'] == 'scmri'):
            df.loc[index, 'AGE'] = data.loc[inx, 'AGE']
            # print("======")
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
df = correct_columnname(df)
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
# data.to_csv('age.csv', index=0)
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

# data = data.drop(['PTEDUCAT_y', 'PTETHCAT_y', 'PTRACCAT_y', 'PTGENDER_y'], axis=1)
# data = data.rename(columns={ 'PTEDUCAT_x':'PTEDUCAT', 'PTETHCAT_x':'PTETHCAT', 'PTRACCAT_x':'PTRACCAT', 'PTGENDER_x':'PTGENDER'})
data = correct_columnname(data)

#基因----有14条数据没有找到apoe信息，默认值为零
APOERES = pd.read_csv('../raw_data/APOERES.csv')
data = pd.merge(data, APOERES, on=['RID'], how='left')
data = correct_columnname(data)
df = copy.deepcopy(data)
for index, row in data.iterrows():

    if (str(row['APGEN1']), str(row['APGEN2'])) in [('3.0', '4.0'), ('4.0', '4.0')]:  # 顺序不能变
        df.loc[index, 'APOE4'] = 1
    else:
        # print(df.loc[index, 'APOE4'].dtype)
        if (str(df.loc[index, 'APOE4']) != '') | (str(row['APGEN1']) != ''):
            df.loc[index, 'APOE4'] = 0

# df = df.rename(columns={'VISCODE_x': 'VISCODE'})
# df.to_csv('our_img_merge.csv', index=0)

#状态信息DX
ADNI_DXSUM_PDXCONV = pd.read_csv('../raw_data/ADNI_DXSUM_PDXCONV.csv')
EXCLUSIO = pd.read_csv('../raw_data/EXCLUSIO.csv')
# df = pd.read_csv('our_img_merge.csv')

df = pd.merge(df, ADNI_DXSUM_PDXCONV, on=['RID', 'VISCODE'], how='left')
df = correct_columnname(df)
data = copy.deepcopy(df)

for index, row in df.iterrows(): #一共有561条数据没有匹配到状态，由于ADNI_DXSUM_PDXCONV表的字段缺失
    # print("row['DXCURREN'].type= ", row['DXCURREN'])
    if row['DXCURREN'] != row['DXCURREN']:  #row['DXCURREN']为空的时候
        # print("into----")

        if row['DX'] != '':
            if row['DX'] == 'CN':
                data.loc[index, 'DXCURREN'] = 1
            elif row['DX'] == 'MCI':
                data.loc[index, 'DXCURREN'] = 2
            elif row['DX'] == 'Dementia':
                data.loc[index, 'DXCURREN'] = 3
        else:
            data.loc[index, 'DXCURREN'] = 0

# data.to_csv('our_img_merge.csv', index=0)

# data = data.reindex(columns=['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER'], fill_value=0)


#诊断信息标志位
def addColumns_diagnosis(data):
    # i = 0
    df = copy.deepcopy(data)
    targetTable = '../raw_data/ADNI_DXSUM_PDXCONV.csv'
    exclusTable = '../raw_data/EXCLUSIO.csv'

    for index, row in data.iterrows():
        if row['Phase'] == 'ADNI 1':
            if row['DXCURREN'] == row['DXCURREN']:
                if str(row['DXCURREN']) == '1.0': # NL healthy control
                    for var in ['MCI', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:  # Note that PD is not included here, since all DXPARK=-4
                        df.loc[index, var] = 0  # 0 means no
                    df.loc[index, 'NC'] = 1      # 1 means yes
                    df.loc[index, 'COG'] = 0
                    df.loc[index, 'ALL'] = 0
                elif str(row['DXCURREN']) == '2.0': # MCI patient
                    for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:  # Note that PD is not included here, since all DXPARK=-4
                        df.loc[index, var] = 0
                    df.loc[index, 'MCI'] = 1
                    df.loc[index, 'COG'] = 1
                    df.loc[index, 'ALL'] = 1
                elif str(row['DXCURREN']) == '3.0': # Dementia patient
                    for var in ['NC', 'MCI']:
                        df.loc[index, var] = 0
                    df.loc[index, 'COG'] = 2
                    df.loc[index, 'ADD'] = 1
                    df.loc[index, 'AD'] = 1
                    df.loc[index, 'DE'] = 1
                    df.loc[index, 'ALL'] = 2
                    print("str(row['DXCURREN']) == '3.0'")

                    if row['DXOTHDEM'] != '-4': # all AD cases has DXOTHDEM = -4, thus other dementia info is unknown
                        print('found AD case with other dementia info')
                else:
                    print(row['DXCURREN']) # no print out, DXCURREN can only take value 1, 2, 3
                if str(row['DXPARK']) != '-4':
                    print('found a case with PD info')  # no print here, turns out all DXPARK=-4
                    df.loc[index, 'PD'] = row['DXPARK']

        if row['Phase'] == 'ADNI 2':
            # print("row['Phase'] == 'ADNI 2'")
            if row['DXCHANGE'] == row['DXCHANGE']:
                # print("row['DXCHANGE']= ", row['DXCHANGE'])
                if str(row['DXCHANGE']) in ['1.0', '7.0', '9.0']:  # NL healthy control
                    for var in ['MCI', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:
                        df.loc[index, var] = 0  # 0 means no
                    df.loc[index, 'NC'] = 1  # 1 means yes
                    df.loc[index, 'COG'] = 0
                    df.loc[index, 'ALL'] = 0
                elif str(row['DXCHANGE']) in ['2.0', '4.0', '8.0']:  # MCI patient
                    for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:
                        df.loc[index, var] = 0
                    df.loc[index, 'MCI'] = 1
                    df.loc[index, 'COG'] = 1
                    df.loc[index, 'ALL'] = 1
                elif str(row['DXCHANGE']) in ['3.0', '5.0', '6.0']:  # Dementia patient
                    df.loc[index, 'NC'] = 0
                    df.loc[index, 'MCI'] = 0
                    df.loc[index, 'DE'] = 1
                    df.loc[index, 'COG'] = 2
                    df.loc[index, 'ADD'] = 0
                    df.loc[index, 'ALL'] = 3
                    if str(row['DXDDUE']) == '1.0':
                        df.loc[index, 'AD'] = 1
                        df.loc[index, 'ADD'] = 1
                        df.loc[index, 'ALL'] = 2
                    elif str(row['DXDDUE']) == '2.0':
                        if str(row['DXODES']) == '1.0': df.loc[index, 'FTD'] = 1
                        if str(row['DXODES']) == '12.0': df.loc[index, 'FTD'] = 1
                        if str(row['DXODES']) == '2.0': df.loc[index, 'PD'] = 1
                        if str(row['DXODES']) == '2.0': df.loc[index, 'PDD'] = 1
                        if str(row['DXODES']) == '9.0': df.loc[index, 'VD'] = 1

        if row['Phase'] == 'ADNI 3':

            if row['DIAGNOSIS'] == row['DIAGNOSIS']:
                if str(row['DIAGNOSIS']) == '1.0':  # NL healthy control
                    # print("row['Phase'] == 'ADNI 3'")
                    for var in ['MCI', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:  # Note that PD is not included here
                        df.loc[index, var] = 0  # 0 means no
                    df.loc[index, 'NC'] = 1  # 1 means yes
                    df.loc[index, 'COG'] = 0
                    df.loc[index, 'ALL'] = 0
                elif str(row['DIAGNOSIS']) == '2.0':  # MCI patient
                    for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:  # Note that PD is not included here, since all DXPARK=-4
                        df.loc[index, var] = 0
                    df.loc[index, 'MCI'] = 1
                    df.loc[index, 'COG'] = 1
                    df.loc[index, 'ALL'] = 1
                elif str(row['DIAGNOSIS']) == '3.0':  # Dementia patient
                    for var in ['NC', 'MCI']:
                        df.loc[index, var] = 0
                    df.loc[index, 'DE'] = 1
                    df.loc[index, 'COG'] = 2
                    df.loc[index, 'ADD'] = 1
                    if str(row['DXDDUE']) == '1.0': # dementia due to AD
                        df.loc[index, 'AD'] = 1
                        df.loc[index, 'ALL'] = 2
                    else: # value 2 means dementia due to ether etiologies
                        pass # turns out all 'DXDDUE'=='1'for dementia cases

        if row['Phase'] == 'ADNI GO':
            print("row['Phase'] == 'ADNI GO'")
            for var in ['NC', 'DE', 'AD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:
                df.loc[index, var] = 0
            df.loc[index, 'MCI'] = 1
            df.loc[index, 'COG'] = 1
            df.loc[index, 'ALL'] = 1
    if str(row['ADD']) == '0.0':
        df.loc[index, 'ALL'] = 3

    exclusTable = pd.read_csv(exclusTable)
    df = pd.merge(df, exclusTable, on=['RID', 'VISCODE'], how='left')
    df = correct_columnname(df)
    data = copy.deepcopy(df)
    # exclusTable = readcsv(exclusTable)
    for index, row in df.iterrows(): # this table is only for ADNI1 screening case, use the table to check whether other neurologic diseases exist
        # print("row['EXNEURO'])= ",row['EXNEURO'])
        if str(row['EXNEURO']) == '0.0': # no other neurologic disease
            data.loc[index, 'PD'] = 0  # including no parkinson
            data.loc[index, 'VD'] = 0
            data.loc[index, 'PDD'] = 0
            # print("======")
        elif str(row['EXNEURO']) == '1.0': # there is other neurologic disease
            # check subtype using DXODES in target table
            print('found other neruologic diseases')
            print(data.loc[index, 'RID'])
    # print(i)
    return df

data = addColumns_diagnosis(data)

#MMSE
def addColumns_mmse(data):

    ADNI_MMSE = pd.read_csv('../raw_data/ADNI_MMSE.csv')
    df = pd.merge(data, ADNI_MMSE, on=['RID', 'VISCODE'], how='left')
    df = correct_columnname(df)
    data = copy.deepcopy(df)
    # data.to_csv('our_img_merge.csv', index=0)
    for index, row in df.iterrows():
        # print("row['MMSCORE']= ", str(row['MMSCORE']))
        if row['MMSE'] != row['MMSE']:
            # print("comming!!!!!!!!!!!!!!!!")
            data.loc[index, 'MMSE'] = row['MMSCORE']
    return data

data = addColumns_mmse(data)

#CDR
def addColumns_cdr(data):

    # print("ooooooooooo")
    CDR = pd.read_csv('../raw_data/CDR.csv')
    data = pd.merge(data, CDR, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)
    old_variables = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE']
    for index, row in data.iterrows():
        sumScore = 0
        for var in old_variables:
            sumScore += float(row[var])
        df.loc[index, 'cdrSum'] = sumScore
        df.loc[index, 'cdr'] = row['CDGLOBAL']

    return df
# data = pd.concat([data, pd.DataFrame(columns=['cdr', 'cdrSum'])], sort=False) #在添加状态位的时候添加了
data = addColumns_cdr(data)

#tesla
def addColumns_tesla(data):
    df = copy.deepcopy(data)
    for index, row in data.iterrows():
        # print("row['FLDSTRENG'].type= ", type(row['FLDSTRENG']))
        # print("row['FLDSTRENG']= ", row['FLDSTRENG'])
        if row['FLDSTRENG'] == row['FLDSTRENG']:
            df.loc[index, 'Tesla'] = row['FLDSTRENG'][:-10]
    return df

data = addColumns_tesla(data)

#ADAS
def addColumns_ADAS(data):
    ADASSCORES = pd.read_csv('../raw_data/ADASSCORES.csv')
    data = pd.merge(data, ADASSCORES, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)
    variables = ['adas_q1', 'adas_q2', 'adas_q3', 'adas_q4', 'adas_q5', 'adas_q6', 'adas_q7', 'adas_q8', 'adas_q9',
                 'adas_q10', 'adas_q11', 'adas_q12', 'adas_q14', 'adas_total11', 'adas_totalmod']
    old_variables = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q14', 'TOTAL11',
                     'TOTALMOD']

    for index, row in data.iterrows():
        for old_var, new_var in zip(old_variables, variables):
            df.loc[index, new_var] = row[old_var]

    #对于ADNI 2, 3, GO
    ADAS_ADNIGO23 = pd.read_csv('../raw_data/ADAS_ADNIGO23.csv')
    data = pd.merge(df, ADAS_ADNIGO23, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)
    variables = ['adas_q1', 'adas_q2', 'adas_q3', 'adas_q4', 'adas_q5', 'adas_q6', 'adas_q7', 'adas_q8', 'adas_q9',
                 'adas_q10', 'adas_q11', 'adas_q12', 'adas_q14', 'adas_total11', 'adas_totalmod']
    old_variables = ['Q1SCORE', 'Q2SCORE', 'Q3SCORE', 'Q4SCORE', 'Q5SCORE', 'Q6SCORE', 'Q7SCORE', 'Q8SCORE',
                     'Q9SCORE', 'Q10SCORE', 'Q11SCORE', 'Q12SCORE', 'Q13SCORE', 'TOTAL13', 'TOTSCORE']

    for index, row in data.iterrows():
        for old_var, new_var in zip(old_variables, variables):
            if (row['Phase'] == 'ADNI 2') | (row['Phase'] == 'ADNI 3') | (row['Phase'] == 'ADNI GO'):
                df.loc[index, new_var] = row[old_var]

    return df

data = addColumns_ADAS(data)

#TrailMaking
def addColumns_TrailMaking(data):
    ITEM = pd.read_csv('../raw_data/ITEM.csv')
    variables = ['trailA', 'trailB']
    old_variables = ['TMT_PtA_Complete', 'TMT_PtB_Complete']
    data = pd.merge(data, ITEM, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)
    for index, row in data.iterrows():
        # print("row['TMT_PtA_Complete'].type= ", type(row['TMT_PtA_Complete']))
        # print("row['TMT_PtA_Complete']= ", row['TMT_PtA_Complete'])
        if row['TMT_PtA_Complete'] == row['TMT_PtA_Complete']:
            if row['TMT_PtA_Complete'] and row['TMT_PtA_Complete'] != 'NULL' and 0 <= int(row['TMT_PtA_Complete']) <= 150:
                df.loc[index, 'trailA'] = row['TMT_PtA_Complete']
            if row['TMT_PtB_Complete'] and row['TMT_PtB_Complete'] != 'NULL' and 0 <= int(row['TMT_PtB_Complete']) <= 300:
                df.loc[index, 'trailB'] = row['TMT_PtB_Complete']
    return df

data = addColumns_TrailMaking(data)

#logicalmemory
def addColumns_logicalmemory(data):
    NEUROBAT = pd.read_csv('../raw_data/NEUROBAT.csv')
    data = pd.merge(data, NEUROBAT, on=['RID', 'VISCODE'], how='left') #['sc', 'bl']后面数据量少要考虑
    data = correct_columnname(data)
    df = copy.deepcopy(data)


    for index, row in data.iterrows():
        if (row['LIMMTOTAL'] == row['LIMMTOTAL']):
            if row['LIMMTOTAL'] and 0 <= int(row['LIMMTOTAL']) <= 25:
                df.loc[index, 'lm_imm'] = row['LIMMTOTAL']
        # print("row['LDELTOTAL'] = ", row['LDELTOTAL'])
        if (row['LDELTOTAL'] == row['LDELTOTAL']):
            if row['LDELTOTAL'] and 0 <= int(row['LDELTOTAL']) <= 25:
                df.loc[index, 'lm_del'] = row['LDELTOTAL']
        else:
            df.loc[index, 'lm_del'] = row['LDELTOTAL_origin']
        if (row['BNTTOTAL'] == row['BNTTOTAL']):
            if row['BNTTOTAL'] and 0 <= int(row['BNTTOTAL']) <= 30:
                df.loc[index, 'boston'] = row['BNTTOTAL']
        if (row['CATANIMSC'] == row['CATANIMSC']):
            if row['CATANIMSC'] and 0 <= int(row['CATANIMSC']) <= 77:
                df.loc[index, 'animal'] = row['CATANIMSC']
        if (row['CATVEGESC'] == row['CATVEGESC']):
            if row['CATVEGESC'] and 0 <= int(row['CATVEGESC']) <= 77:
                df.loc[index, 'vege'] = row['CATVEGESC']
        if (row['DSPANBAC'] == row['DSPANBAC']):
            if row['DSPANBAC'] and 0 <= int(row['DSPANBAC']) <= 12:
                df.loc[index, 'digitB'] = row['DSPANBAC']
        if (row['DSPANFOR'] == row['DSPANFOR']):
            if row['DSPANFOR'] and 0 <= int(row['DSPANFOR']) <= 12:
                df.loc[index, 'digitF'] = row['DSPANFOR']
        if (row['DSPANBLTH'] == row['DSPANBLTH']):
            if row['DSPANBLTH'] and 0 <= int(row['DSPANBLTH']) <= 8:
                df.loc[index, 'digitBL'] = row['DSPANBLTH']
        if (row['DSPANFLTH'] == row['DSPANFLTH']):
            if row['DSPANFLTH'] and 0 <= int(row['DSPANFLTH']) <= 8:
                df.loc[index, 'digitFL'] = row['DSPANFLTH']
    return df

# data = data.drop(['LDELTOTAL', 'LDELTOTAL_BL'], axis=1)
data = data.rename(columns={'LDELTOTAL': 'LDELTOTAL_origin', 'LDELTOTAL_BL': 'LDELTOTAL_BL_origin'})
data = addColumns_logicalmemory(data)

#moca
def addColumns_Moca(data):
    MOCA = pd.read_csv('../raw_data/ADNIMERGE.csv')
    data = pd.merge(data, MOCA, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)

    for index, row in data.iterrows():
        if row['MOCA'] == row['MOCA']:
            df.loc[index, 'moca'] = row['MOCA']
        else:
            df.loc[index, 'moca'] = row['MOCA_origin']

    #对于ADNI2 ,3 ,GO
    ADNIGO_MOCA = pd.read_csv('../raw_data/MOCA.csv')
    data = pd.merge(df, ADNIGO_MOCA, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)

    for index, row in data.iterrows():
        if row['moca'] == row['moca']:
            df.loc[index, 'moca'] = row['moca']
        else:
            df.loc[index, 'moca'] = row['MOCA_origin']


    return df

# data = data.drop(['MOCA'], axis=1)
data = data.rename(columns={'MOCA': 'MOCA_origin'})
data = addColumns_Moca(data)

#NPIQ
def addColumns_NPIQ(data):
    NPIQ = pd.read_csv('../raw_data/NPIQ.csv')
    data = pd.merge(data, NPIQ, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)
    variables = ['npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD',
                 'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN',
                 'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP']
    old_variables = ['NPIA', 'NPIB', 'NPIC', 'NPID',
                     'NPIE', 'NPIF', 'NPIG', 'NPIH',
                     'NPII', 'NPIJ', 'NPIK', 'NPIL']

    for index, row in data.iterrows():
        for old_var, new_var in zip(old_variables, variables):
            # print("row[old_var]= ", row[old_var])
            if row[old_var] == row[old_var]:
                if str(row[old_var]) == '0.0':
                    df.loc[index, new_var] = '0.0'
                    # print("---------------------------")
                elif str(row[old_var]) == '1.0':
                    # df.loc[index, new_var] = row[sev_var]
                    df.loc[index, new_var] = row[old_var + "SEV"]

    return df

data =  addColumns_NPIQ(data)

def addColumns_NPI(data):
    NPI = pd.read_csv('../raw_data/NPI.csv')
    data = pd.merge(data, NPI, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)
    variables = ['npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD',
                 'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN',
                 'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP']
    old_variables = ['NPIA', 'NPIB', 'NPIC', 'NPID',
                     'NPIE', 'NPIF', 'NPIG', 'NPIH',
                     'NPII', 'NPIJ', 'NPIK', 'NPIL']
    sev_variables = ['NPIA10B', 'NPIB8B', 'NPIC9B', 'NPID9B',
                     'NPIE8A', 'NPIF8B', 'NPIG9B', 'NPIH8B',
                     'NPII8B', 'NPIJ8B', 'NPIK9B', 'NPIL9B']

    for index, row in data.iterrows():
        for old_var, new_var, sev_var in zip(old_variables, variables, sev_variables):
            # print("row[old_var]= ", row[old_var])
            # print("---------------------------")
            # if row[old_var] == row[old_var]:
            if str(row[old_var]) == '0.0':
                df.loc[index, new_var] = '0.0'

            elif str(row[old_var]) == '1.0':
                df.loc[index, new_var] = row[sev_var]

    return df

data = data.drop(['NPIA', 'NPIB', 'NPIC', 'NPID',
                     'NPIE', 'NPIF', 'NPIG', 'NPIH',
                     'NPII', 'NPIJ', 'NPIK', 'NPIL'], axis=1)
data =  addColumns_NPI(data)

#FAQ
def addColumns_FAQ(data):
    FAQ = pd.read_csv('../raw_data/FAQ.csv')
    data = pd.merge(data, FAQ, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)
    variables = ['faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
                 'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL']
    old_variables = ['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG',
                     'FAQMEAL', 'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']

    for index, row in data.iterrows():
        for old_var, new_var in zip(old_variables, variables):
            if row[old_var] == row[old_var]:
                df.loc[index, new_var] = row[old_var]
    return df

data = addColumns_FAQ(data)


#MEDHIST
def addColumns_medhist(data):
    MEDHIST = pd.read_csv('../raw_data/MEDHIST.csv')
    data = pd.merge(data, MEDHIST, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)
    variables = ['his_CVHATT', 'his_PSYCDIS', 'his_Alcohol', 'his_SMOKYRS', 'his_PACKSPER']
    old_variables = ['MH4CARD', 'MHPSYCH', 'MH14ALCH', 'MH16BSMOK', 'MH16ASMOK']

    for index, row in data.iterrows():
        for old_var, new_var in zip(old_variables, variables):
            if row[old_var] == row[old_var]:
                if new_var == 'his_SMOKYRS':
                    if str(row[old_var]) != '-4.0':
                        df.loc[index, new_var] = row[old_var]
                elif new_var == 'his_PACKSPER':
                    if str(row[old_var]) != '-4.0':
                        df.loc[index, new_var] = float(row[old_var]) * 365
                else:
                    df.loc[index, new_var] = row[old_var]
    return df

data = addColumns_medhist(data)

#GDSCALE
def addColumns_GDS(data):
    GDSCALE = pd.read_csv('../raw_data/GDSCALE.csv')
    data = pd.merge(data, GDSCALE, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)

    for index, row in data.iterrows():
        if row['GDTOTAL'] == row['GDTOTAL']:
            if 0 <= int(row['GDTOTAL']) <= 15:
                df.loc[index, 'gds'] = row['GDTOTAL']
    return df

data = addColumns_GDS(data)

#FHQ
def addColumns_FHQ(data):
    FHQ = pd.read_csv('../raw_data/FHQ.csv')
    data = pd.merge(data, FHQ, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)

    for index, row in data.iterrows():
        if (row['FHQMOM'] == row['FHQMOM']) | (row['FHQDAD'] == row['FHQDAD']) | (row['FHQSIB'] == row['FHQSIB']):
            if (str(row['FHQMOM']) == '1.0' or str(row['FHQDAD']) == '1.0' or str(row['FHQSIB']) == '1.0'):
                df.loc[index, 'his_NACCFAM'] = '1'
            else:
                df.loc[index, 'his_NACCFAM'] = '0'

    #对于ADNI3
    FAMXHPAR = pd.read_csv('../raw_data/FAMXHPAR.csv')
    data = pd.merge(df, FAMXHPAR, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)

    for index, row in data.iterrows():
        if row['Phase'] == 'ADNI 3':
            if (row['MOTHDEM'] == row['MOTHDEM']) | (row['FATHDEM'] == row['FATHDEM']):
                if str(row['MOTHDEM']) == '1.0' or (row['FATHDEM'] == row['FATHDEM']):
                    df.loc[index, 'his_NACCFAM'] = '1'
                else:
                    df.loc[index, 'his_NACCFAM'] = '0'

    FAMXHSIB = pd.read_csv('../raw_data/FAMXHSIB.csv')
    data = pd.merge(df, FAMXHSIB, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)

    for index, row in data.iterrows():
        if row['Phase'] == 'ADNI 3':
            if (row['his_NACCFAM'] == row['his_NACCFAM']) | (row['SIBDEMENT'] == row['SIBDEMENT']):
                if str(row['his_NACCFAM']) == '1.0':
                    continue
                if str(row['SIBDEMENT']) == '1.0':
                    df.loc[index, 'his_NACCFAM'] = '1'
                else:
                    df.loc[index, 'his_NACCFAM'] = '0'


    return df

data = addColumns_FHQ(data)

#MODHACH
def addColumns_modhach(data):
    MODHACH = pd.read_csv('../raw_data/MODHACH.csv')
    data = pd.merge(data, MODHACH, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)
    variables = ['his_CBSTROKE', 'his_HYPERTEN']
    old_variables = ['HMSTROKE', 'HMHYPERT']

    for index, row in data.iterrows():
        for old_var, new_var in zip(old_variables, variables):
            df.loc[index, new_var] = row[old_var]
    return df

data = addColumns_modhach(data)

#his_DEPOTHR
def addColumns_dep(data):
    ADNI_DXSUM_PDXCONV = pd.read_csv('../raw_data/ADNI_DXSUM_PDXCONV.csv')
    data = pd.merge(data, ADNI_DXSUM_PDXCONV, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)

    for index, row in data.iterrows():
        if str(row['DXNODEP']) == '1.0':
            df.loc[index, 'his_DEPOTHR'] = '1'
        else:
            df.loc[index, 'his_DEPOTHR'] = '0'

    #对ADNI 2, 3, GO
    BLCHANGE = pd.read_csv('../raw_data/BLCHANGE.csv')
    data = pd.merge(df, BLCHANGE, on=['RID', 'VISCODE'], how='left')
    data = correct_columnname(data)
    df = copy.deepcopy(data)

    for index, row in data.iterrows():
        if row['Phase'] != 'ADNI 1':
            df.loc[index, 'his_DEPOTHR'] = row['BCDEPRES']

    return df

data = data.drop(['DXNODEP'], axis=1)
data = addColumns_dep(data)

data.to_csv('our_img_merge.csv', index=0)


#按照模型的输入，筛选字段


# data.to_csv('our_img_merge.csv', index=0)
data = data.rename(columns={'Phase':'path','AGE': 'age', 'PTGENDER': 'gender', 'PTEDUCAT': 'education', 'MMSE': 'mmse', 'his_Alcohol':'his_ALCOHOL'})

features = [
                "RID","VISCODE", "path","filename",'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER', "age", "gender", "education", "trailA", "trailB", "boston", "digitB", "digitBL", "digitF", "digitFL",
                "animal", "gds", "lm_imm", "lm_del", "mmse", "npiq_DEL", "npiq_HALL", "npiq_AGIT", "npiq_DEPD", "npiq_ANX",
                "npiq_ELAT", "npiq_APA", "npiq_DISN", "npiq_IRR", "npiq_MOT", "npiq_NITE", "npiq_APP", "faq_BILLS", "faq_TAXES", "faq_SHOPPING",
                "faq_GAMES", "faq_STOVE", "faq_MEALPREP", "faq_EVENTS", "faq_PAYATTN", "faq_REMDATES", "faq_TRAVEL", "his_NACCFAM", "his_CVHATT", "his_CVAFIB",
                "his_CVANGIO", "his_CVBYPASS", "his_CVPACE", "his_CVCHF", "his_CVOTHR", "his_CBSTROKE", "his_CBTIA", "his_SEIZURES", "his_TBI", "his_HYPERTEN",
                "his_HYPERCHO", "his_DIABETES", "his_B12DEF", "his_THYROID", "his_INCONTU", "his_INCONTF", "his_DEP2YRS", "his_DEPOTHR", "his_PSYCDIS", "his_ALCOHOL",
                "his_TOBAC100", "his_SMOKYRS", "his_PACKSPER", "his_ABUSOTHR", "COG_score", "ADD_score"
              ]
# features = [
#                 "Phase","filename","RID","VISCODE", 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER',"age", "gender", "education", "trailA", "trailB", "boston", "digitB", "digitBL", "digitF", "digitFL",
#                 "animal", "gds", "lm_imm", "lm_del", "mmse", "npiq_DEL", "npiq_HALL", "npiq_AGIT", "npiq_DEPD", "npiq_ANX",
#                 "npiq_ELAT", "npiq_APA", "npiq_DISN", "npiq_IRR", "npiq_MOT", "npiq_NITE", "npiq_APP", "faq_BILLS", "faq_TAXES", "faq_SHOPPING",
#                 "faq_GAMES", "faq_STOVE", "faq_MEALPREP", "faq_EVENTS", "faq_PAYATTN", "faq_REMDATES", "faq_TRAVEL", "his_NACCFAM", "his_CVHATT",
#                    "his_CBSTROKE", "his_HYPERTEN",
#                 "his_DEPOTHR", "his_PSYCDIS", "his_ALCOHOL",
#                 "his_SMOKYRS", "his_PACKSPER"
#             ]
#, "his_CVAFIB","his_CVANGIO","his_CVBYPASS","his_CVPACE", "his_CVCHF","his_CVOTHR", "his_CBTIA","his_SEIZURES",
#"his_TBI", "his_HYPERCHO", "his_DIABETES", "his_B12DEF", "his_THYROID", "his_INCONTU", "his_INCONTF", "his_DEP2YRS",
#"his_TOBAC100", "his_ABUSOTHR", , "COG_score", "ADD_score"

data = data.loc[:, features]

#删除NC, ADD状态为为空的字段561
df = copy.deepcopy(data)
i = 0
for index, row in df.iterrows():
    if row['NC'] != row['NC']:
        data = data.drop(index=df.index[index])
        i += 1
print("drop i= ", i)

# data['AD'] = data['AD'].astype('int64')
# data['ALL'] = data['ALL'].astype('int64')
# data['age'] = data['age'].astype('int64')
# data['education'] = data['education'].astype('int64')
# data['mmse'] = data['mmse'].astype('int64')
# data['his_NACCFAM'] = data['his_NACCFAM'].astype('int64')
# data['his_CBSTROKE'] = data['his_CBSTROKE'].astype('int64')
# data['his_HYPERTEN'] = data['his_HYPERTEN'].astype('int64')

data.to_csv('ADNI.csv', index=0) #817
