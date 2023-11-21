import pandas as pd
import numpy as np
import copy as cp
from sklearn.model_selection import train_test_split
import random
import math
from sklearn.impute import SimpleImputer



def addColumns_demograph(train_data):
    train_data['VISCODE_No'] = train_data['VISCODE']
    train_data.loc[train_data['VISCODE_No'] == 'bl', 'VISCODE_No'] = 'm0'
    train_data["VISCODE_No"] = train_data["VISCODE_No"].str.replace("m", "").astype("int32")
    train_data.sort_values(['RID', 'VISCODE_No'], inplace=True, ascending=True)  # 按多列排序
    train_data.dropna(subset=['AGE', 'APOE4', 'DX'], inplace=True)

    deep_merge_data = cp.deepcopy(train_data)
    deep_merge_data.drop_duplicates(subset=['RID'], inplace=True)

    for index, row in train_data.iterrows():
        RID = int(row['RID'])
        viscode = int(row['VISCODE_No'])
        if viscode == 0:
            continue
        else:
            AGE_bl = float(deep_merge_data[deep_merge_data["RID"] == RID]['AGE'])
            EXAMDATE_bl = str(deep_merge_data[deep_merge_data["RID"] == RID]['EXAMDATE'].values.tolist()[0])
            EXAMDATEs_bl = str(EXAMDATE_bl).split('-')
            Year_bl, Month_bl, Day_bl = int(EXAMDATEs_bl[0]), int(EXAMDATEs_bl[1]), int(EXAMDATEs_bl[2])

            EXAMDATE = row['EXAMDATE']
            EXAMDATEs = str(EXAMDATE).split('-')
            Year, Month, Day = int(EXAMDATEs[0]), int(EXAMDATEs[1]), int(EXAMDATEs[2])

            if Month >= Month_bl:
                Month -= Month_bl
            else:
                Year -= 1
                Month += 12
                Month -= Month_bl
            if Month != 0:
                Month = Month / 12
            Year -= Year_bl
            AGE = round(Year + Month + AGE_bl, 1)
            train_data.loc[index, 'AGE'] = AGE

    train_data["PTGENDER"] = train_data["PTGENDER"].str.replace("Male", '0')
    train_data["PTGENDER"] = train_data["PTGENDER"].str.replace("Female", '1')
    train_data["PTGENDER"] = train_data["PTGENDER"].astype("int32")

    train_data["PTETHCAT"] = train_data["PTETHCAT"].str.replace("Not Hisp/Latino", '2')
    train_data["PTETHCAT"] = train_data["PTETHCAT"].str.replace("Hisp/Latino", '1')
    train_data["PTETHCAT"] = train_data["PTETHCAT"].str.replace("Unknown", '3')

    train_data["PTRACCAT"] = train_data["PTRACCAT"].str.replace("Am Indian/Alaskan", '1')
    train_data["PTRACCAT"] = train_data["PTRACCAT"].str.replace("Asian", '2')
    train_data["PTRACCAT"] = train_data["PTRACCAT"].str.replace("Hawaiian/Other PI", '3')
    train_data["PTRACCAT"] = train_data["PTRACCAT"].str.replace("Black", '4')
    train_data["PTRACCAT"] = train_data["PTRACCAT"].str.replace("White", '5')
    train_data["PTRACCAT"] = train_data["PTRACCAT"].str.replace("More than one", '6')
    train_data["PTRACCAT"] = train_data["PTRACCAT"].str.replace("Unknown", '7')

    train_data["PTMARRY"] = train_data["PTMARRY"].str.replace("Never married", '3')
    train_data["PTMARRY"] = train_data["PTMARRY"].str.replace("Married", '1')
    train_data["PTMARRY"] = train_data["PTMARRY"].str.replace("Divorced", '2')
    train_data["PTMARRY"] = train_data["PTMARRY"].str.replace("Widowed", '4')
    train_data["PTMARRY"] = train_data["PTMARRY"].str.replace("Unknown", '5')

    return train_data

def addColumns_diagnosis(data):
    base_columns = data.columns.values.tolist()
    columns = ['NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER']
    base_columns += columns
    targetTable = pd.read_csv('./raw_data/ADNI_DXSUM_PDXCONV.csv')
    targetTable.dropna(subset=['DXCURREN'], inplace=True)
    targetTable.drop(columns=['VISCODE'], inplace=True)
    targetTable.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    targetTable = targetTable.loc[:, ['RID', 'VISCODE', 'DXCURREN', 'DXOTHDEM', 'DXPARK', 'DXODES']]

    merge_data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')
    deep_merge_data = cp.deepcopy(merge_data)
    deep_merge_data = pd.concat([deep_merge_data, pd.DataFrame(columns=['NC', 'MCI', 'AD', 'PDD'])], sort=False)

    exclusTable = pd.read_csv('./raw_data/EXCLUSIO.csv')
    exclusTable = exclusTable[exclusTable["EXNEURO"] == 1]
    RIDs_exclusTable = exclusTable.loc[:, 'RID'].values.tolist()

    for index, row in merge_data.iterrows():
        if row['DXCURREN'] == 1: # NL healthy control
            for var in ['MCI', 'DE', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']: # Note that PD is not included here, since all DXPARK=-4
                deep_merge_data.loc[index, var] = 0   # 0 means no
            deep_merge_data.loc[index, 'NC'] = 1      # 1 means yes
            deep_merge_data.loc[index, 'COG'] = 0
            deep_merge_data.loc[index, 'ALL'] = 0
        elif row['DXCURREN'] == 2: # MCI patient
            for var in ['NC', 'DE', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']: # Note that PD is not included here, since all DXPARK=-4
                deep_merge_data.loc[index, var] = 0
            deep_merge_data.loc[index, 'MCI'] = 1
            deep_merge_data.loc[index, 'COG'] = 1
            deep_merge_data.loc[index, 'ALL'] = 1
        elif row['DXCURREN'] == 3: # Dementia patient
            for var in ['NC', 'MCI', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']:
                deep_merge_data.loc[index, var] = 0
            deep_merge_data.loc[index, 'COG'] = 2
            deep_merge_data.loc[index, 'ADD'] = 1
            deep_merge_data.loc[index, 'AD'] = 1
            deep_merge_data.loc[index, 'DE'] = 1
            deep_merge_data.loc[index, 'ALL'] = 2
        else:
            print(row['DXCURREN']) # no print out, DXCURREN can only take value 1, 2, 3

        if row['DXPARK'] != -4:
            print('found a case with PD info')  # no print here, turns out all DXPARK=-4
            deep_merge_data.loc[index, 'PD'] = row['DXPARK']

        if row['RID'] in RIDs_exclusTable:
            print('found other neruologic diseases')
            deep_merge_data.loc[index, 'PD'] = 1
        else:
            deep_merge_data.loc[index, 'PD'] = 0  # including no parkinson
            deep_merge_data.loc[index, 'VD'] = 0
            deep_merge_data.loc[index, 'PDD'] = 0

    deep_merge_data = deep_merge_data.drop(deep_merge_data[(deep_merge_data.PD == 1)].index)
    deep_merge_data.loc[deep_merge_data['DX'] == 'Dementia', ['AD', 'DE', 'ADD']] = 1
    deep_merge_data.loc[deep_merge_data['DX'] == 'Dementia', ['NC', 'MCI', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'OTHER']] = 0
    deep_merge_data.loc[deep_merge_data['DX'] == 'Dementia', ['COG', 'ALL']] = 2

    deep_merge_data.loc[deep_merge_data['DX'] == 'CN', ['NC']] = 1
    deep_merge_data.loc[deep_merge_data['DX'] == 'CN', ['AD', 'MCI', 'COG', 'ALL', 'DE', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']] = 0

    deep_merge_data.loc[deep_merge_data['DX'] == 'MCI', ['MCI', 'COG', 'ALL']] = 1
    deep_merge_data.loc[deep_merge_data['DX'] == 'MCI', ['AD', 'NC', 'DE', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER']] = 0
    deep_merge_data = deep_merge_data.loc[:, base_columns]

    deep_merge_data["DX"] = deep_merge_data["DX"].str.replace("CN", "0")
    deep_merge_data["DX"] = deep_merge_data["DX"].str.replace("MCI", "1")
    deep_merge_data["DX"] = deep_merge_data["DX"].str.replace("Dementia", "2")

    return deep_merge_data.loc[:, base_columns]

def addColumns_ADAS(data):
    data.drop(columns=['ADAS11', 'ADAS13', 'ADASQ4'], inplace=True)
    base_columns = data.columns.values.tolist()
    columns = ['adas_q1','adas_q2','adas_q3','adas_q4','adas_q5','adas_q6','adas_q7','adas_q8','adas_q9','adas_q10',
               'adas_q11','adas_q12','adas_q14', 'adas_total11', 'adas_totalmod']
    base_columns += columns
    targetTable1 = './raw_data/ADASSCORES.csv'
    targetTable1 = pd.read_csv(targetTable1)
    targetTable1.rename(columns={"Q1": 'adas_q1',"Q2": 'adas_q2',"Q3": 'adas_q3',"Q4": 'adas_q4',"Q5": 'adas_q5',"Q6": 'adas_q6',
                                 "Q7": 'adas_q7',"Q8": 'adas_q8',"Q9": 'adas_q9',"Q10": 'adas_q10',"Q11": 'adas_q11',"Q12": 'adas_q12',
                                 "Q14": 'adas_q14',"TOTAL11": 'adas_total11',"TOTALMOD": 'adas_totalmod'}, inplace=True)
    targetTable1 = targetTable1.loc[:, ['RID', 'VISCODE', 'adas_q1','adas_q2','adas_q3','adas_q4',
                                        'adas_q5','adas_q6','adas_q7','adas_q8','adas_q9','adas_q10',
                                        'adas_q11','adas_q12','adas_q14', 'adas_total11', 'adas_totalmod']]

    targetTable23GO = './raw_data/ADAS_ADNIGO23.csv'
    targetTable23GO = pd.read_csv(targetTable23GO)
    targetTable23GO.drop(columns=['VISCODE'], inplace=True)
    targetTable23GO.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable23GO.rename(
        columns={"Q1SCORE": 'adas_q1', "Q2SCORE": 'adas_q2', "Q3SCORE": 'adas_q3', "Q4SCORE": 'adas_q4', "Q5SCORE": 'adas_q5',
                 "Q6SCORE": 'adas_q6',"Q7SCORE": 'adas_q7', "Q8SCORE": 'adas_q8', "Q9SCORE": 'adas_q9', "Q10SCORE": 'adas_q10',
                 "Q11SCORE": 'adas_q11',"Q12SCORE": 'adas_q12', "Q13SCORE": 'adas_q14', "TOTSCORE": 'adas_total11', "TOTAL13": 'adas_totalmod'}, inplace=True)
    targetTable23GO = targetTable23GO.loc[:, ['RID', 'VISCODE', 'adas_q1','adas_q2','adas_q3','adas_q4',
                                        'adas_q5','adas_q6','adas_q7','adas_q8','adas_q9','adas_q10',
                                        'adas_q11','adas_q12','adas_q14', 'adas_total11', 'adas_totalmod']]
    targetTable = pd.concat([targetTable1, targetTable23GO], axis=0)
    targetTable.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)

    merge_data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')
    merge_data = merge_data.loc[:, base_columns]
    return merge_data.loc[:, base_columns]

def addColumns_logicalmemory(data):
    data.rename(columns={'LDELTOTAL': 'lm_del'}, inplace=True)
    base_columns = data.columns.values.tolist()
    columns = ['lm_imm', 'boston', 'animal', 'vege', 'digitB', 'digitBL', 'digitF', 'digitFL'] #, 'lm_del'
    base_columns += columns
    old_variables = ['RID', 'VISCODE', 'LIMMTOTAL', 'LDELTOTAL', 'BNTTOTAL', 'CATANIMSC', 'CATVEGESC',
                     'DSPANBAC', 'DSPANFOR', 'DSPANBLTH', 'DSPANFLTH']
    targetTable = './raw_data/NEUROBAT.csv'
    targetTable = pd.read_csv(targetTable, low_memory=False)
    targetTable.drop(columns=['VISCODE'], inplace=True)
    targetTable.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable = targetTable.loc[:, old_variables]
    targetTable.loc[targetTable['VISCODE'] == 'sc', 'VISCODE'] = 'bl'
    targetTable.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)

    merge_data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')
    deep_merge_data = cp.deepcopy(merge_data)
    deep_merge_data = pd.concat([deep_merge_data, pd.DataFrame(columns=columns)], sort=False)
    for index, row in merge_data.iterrows():
        if row['LIMMTOTAL'] != row['LIMMTOTAL']:
            row['LIMMTOTAL'] = 0
        if 0 <= int(row['LIMMTOTAL']) <= 25:
            deep_merge_data.loc[index, 'lm_imm'] = row['LIMMTOTAL']
        if row['BNTTOTAL'] != row['BNTTOTAL']:
            row['BNTTOTAL'] = 0
        if 0 <= int(row['BNTTOTAL']) <= 30:
            deep_merge_data.loc[index, 'boston'] = row['BNTTOTAL']
        if row['CATANIMSC'] != row['CATANIMSC']:
            row['CATANIMSC'] = 0
        if 0 <= int(row['CATANIMSC']) <= 77:
            deep_merge_data.loc[index, 'animal'] = row['CATANIMSC']
        if row['CATVEGESC'] != row['CATVEGESC']:
            row['CATVEGESC'] = 0
        if 0 <= int(row['CATVEGESC']) <= 77:
            deep_merge_data.loc[index, 'vege'] = row['CATVEGESC']
        if row['DSPANBAC'] != row['DSPANBAC']:
            row['DSPANBAC'] = 0
        if 0 <= int(row['DSPANBAC']) <= 12:
            deep_merge_data.loc[index, 'digitB'] = row['DSPANBAC']
        if row['DSPANFOR'] != row['DSPANFOR']:
            row['DSPANFOR'] = 0
        if 0 <= int(row['DSPANFOR']) <= 12:
            deep_merge_data.loc[index, 'digitF'] = row['DSPANFOR']
        if row['DSPANBLTH'] != row['DSPANBLTH']:
            row['DSPANBLTH'] = 0
        if 0 <= int(row['DSPANBLTH']) <= 8:
            deep_merge_data.loc[index, 'digitBL'] = row['DSPANBLTH']
        if row['DSPANFLTH'] != row['DSPANFLTH']:
            row['DSPANFLTH'] = 0
        if 0 <= int(row['DSPANFLTH']) <= 8:
            deep_merge_data.loc[index, 'digitFL'] = row['DSPANFLTH']
    deep_merge_data = deep_merge_data.loc[:, base_columns]
    return deep_merge_data.loc[:, base_columns]

def addColumns_Moca(data):
    data.rename(columns={'MOCA': 'moca'}, inplace=True)
    return data

def addColumns_NPIQ(data):
    base_columns = data.columns.values.tolist()
    variables11 = ['npiq_DEL', 'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD',
     'npiq_ANX', 'npiq_ELAT', 'npiq_APA', 'npiq_DISN',
     'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP']
    variables = ['NPIASEV', 'NPIBSEV', 'NPICSEV', 'NPIDSEV',
                    'NPIESEV', 'NPIFSEV', 'NPIGSEV', 'NPIHSEV',
                    'NPIISEV', 'NPIJSEV', 'NPIKSEV', 'NPILSEV']
    old_variables = ['NPIA', 'NPIB', 'NPIC', 'NPID',
                    'NPIE', 'NPIF', 'NPIG', 'NPIH',
                    'NPII', 'NPIJ', 'NPIK', 'NPIL']
    base_columns += variables11
    targetTable = './raw_data/NPIQ.csv'
    targetTable = pd.read_csv(targetTable)
    targetTable = targetTable.loc[:, old_variables + variables + ['RID', 'VISCODE']]
    targetTable.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)

    merge_data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')
    deep_merge_data = cp.deepcopy(merge_data)
    deep_merge_data = pd.concat(
        [deep_merge_data, pd.DataFrame(columns=variables)], sort=False)
    for index, row in merge_data.iterrows():
        for old_var, new_var in zip(old_variables, variables):
            if (row[old_var] == row[old_var]) and str(row[old_var]) == '0.0':
                deep_merge_data.loc[index, new_var] = '0'
            elif (row[old_var] == row[old_var]) and str(row[old_var]) == '1.0':
                deep_merge_data.loc[index, new_var] = row[old_var+"SEV"]

    deep_merge_data.rename(columns={'NPIASEV':'npiq_DEL', 'NPIBSEV':'npiq_HALL', 'NPICSEV':'npiq_AGIT', 'NPIDSEV':'npiq_DEPD',
                 'NPIESEV':'npiq_ANX', 'NPIFSEV':'npiq_ELAT', 'NPIGSEV':'npiq_APA', 'NPIHSEV':'npiq_DISN',
                 'NPIISEV':'npiq_IRR', 'NPIJSEV':'npiq_MOT', 'NPIKSEV':'npiq_NITE', 'NPILSEV':'npiq_APP'}, inplace=True)
    deep_merge_data = deep_merge_data.loc[:, base_columns]
    return deep_merge_data.loc[:, base_columns]

def addColumns_FAQ(data):
    base_columns = data.columns.values.tolist()
    variables = ['faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
                 'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL']
    base_columns += variables
    targetTable = './raw_data/FAQ.csv'
    targetTable = pd.read_csv(targetTable)
    targetTable.drop(columns=['VISCODE'], inplace=True)
    targetTable.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    targetTable.rename(
        columns={'FAQFINAN': 'faq_BILLS', 'FAQFORM': 'faq_TAXES', 'FAQSHOP': 'faq_SHOPPING', 'FAQGAME': 'faq_GAMES',
                 'FAQBEVG': 'faq_STOVE',
                 'FAQMEAL': 'faq_MEALPREP', 'FAQEVENT': 'faq_EVENTS', 'FAQTV': 'faq_PAYATTN', 'FAQREM': 'faq_REMDATES',
                 'FAQTRAVL': 'faq_TRAVEL'}, inplace=True)
    targetTable = targetTable.loc[:, variables + ['RID', 'VISCODE']]
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)

    merge_data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')

    return merge_data.loc[:, base_columns]

def addColumns_GDS(data):
    base_columns = data.columns.values.tolist()
    base_columns += ['gds']
    targetTable = './raw_data/GDSCALE.csv'
    targetTable = pd.read_csv(targetTable)
    targetTable.drop(columns=['VISCODE'], inplace=True)
    targetTable.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable = targetTable.loc[:, ['RID', 'VISCODE', 'GDTOTAL']]
    targetTable.rename(columns={'GDTOTAL': 'gds'}, inplace=True)
    targetTable.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)

    merge_data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')
    return merge_data.loc[:, base_columns]

def addColumns_medhist(data):
    base_columns = data.columns.values.tolist()
    variables = ['his_CVHATT', 'his_PSYCDIS', 'his_ALCOHOL', 'his_SMOKYRS', 'his_PACKSPER']
    old_variables = ['MH4CARD', 'MHPSYCH', 'MH14ALCH', 'MH16BSMOK', 'MH16ASMOK']
    base_columns += variables
    targetTable = './raw_data/MEDHIST.csv'
    targetTable = pd.read_csv(targetTable)
    targetTable.drop(columns=['VISCODE'], inplace=True)
    targetTable.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable.loc[targetTable['VISCODE'] == 'sc', 'VISCODE'] = 'bl'
    targetTable = targetTable.loc[:, ['RID', 'VISCODE'] + old_variables]
    targetTable.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)

    merge_data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')
    deep_merge_data = cp.deepcopy(merge_data)
    deep_merge_data = pd.concat([deep_merge_data, pd.DataFrame(columns=variables)], sort=False)

    for index, row in merge_data.iterrows():
        for old_var, new_var in zip(old_variables, variables):
            if new_var == 'his_SMOKYRS':
                if (row[old_var] == row[old_var]) and str(row[old_var])[0] != '-':
                    deep_merge_data.loc[index, new_var] = row[old_var]
            elif new_var == 'his_PACKSPER':
                if (row[old_var] == row[old_var]) and str(row[old_var]) != '-4.0':
                    deep_merge_data.loc[index, new_var] = float(row[old_var]) * 365
            else:
                deep_merge_data.loc[index, new_var] = row[old_var]

    deep_merge_data = deep_merge_data.loc[:, base_columns]

    return deep_merge_data.loc[:, base_columns]

def addColumns_FHQ(data):
    base_columns = data.columns.values.tolist()
    variables = ['his_NACCFAM']
    base_columns += variables
    targetTable12GO = './raw_data/FHQ.csv'
    targetTable12GO = pd.read_csv(targetTable12GO)
    targetTable12GO = targetTable12GO.loc[:, ['RID', 'FHQMOM', 'FHQDAD', 'FHQSIB']]

    targetTable3 = './raw_data/FAMXHPAR.csv'
    targetTable3 = pd.read_csv(targetTable3)
    targetTable3 = targetTable3.loc[:, ['RID', 'MOTHDEM', 'FATHDEM']]

    targetTable3B = './raw_data/FAMXHSIB.csv'
    targetTable3B = pd.read_csv(targetTable3B)
    targetTable3B = targetTable3B.loc[:, ['RID', 'SIBDEMENT']]
    targetTable3B.rename(columns={'SIBDEMENT': 'FHQSIB'}, inplace=True)

    targetTable3 = pd.merge(targetTable3, targetTable3B, on=['RID'], how='left')
    targetTable3.rename(columns={'MOTHDEM': 'FHQMOM', 'FATHDEM': 'FHQDAD'}, inplace=True)
    targetTable = pd.concat([targetTable12GO, targetTable3], axis=0)
    targetTable.drop_duplicates(subset=['RID'], inplace=True, keep='last')
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)

    merge_data = pd.merge(data, targetTable, on=['RID'], how='left')
    deep_merge_data = cp.deepcopy(merge_data)
    deep_merge_data = pd.concat([deep_merge_data, pd.DataFrame(columns=variables)], sort=False)
    for index, row in merge_data.iterrows():
        if (str(row['FHQMOM'])=='1.0' or str(row['FHQDAD'])=='1.0' or str(row['FHQSIB'])=='1.0'):
            deep_merge_data.loc[index, 'his_NACCFAM'] = '1'
        else:
            deep_merge_data.loc[index, 'his_NACCFAM'] = '0'

    deep_merge_data = deep_merge_data.loc[:, base_columns]
    return deep_merge_data.loc[:, base_columns]

def addColumns_modhach(data):
    base_columns = data.columns.values.tolist()
    variables = ['his_CBSTROKE', 'his_HYPERTEN']
    base_columns += variables
    targetTable = './raw_data/MODHACH.csv'
    targetTable = pd.read_csv(targetTable)
    targetTable.drop(columns=['VISCODE'], inplace=True)
    targetTable.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable.loc[targetTable['VISCODE'] == 'sc', 'VISCODE'] = 'bl'
    targetTable.drop(targetTable[targetTable['VISCODE'] == 'f'].index, inplace=True)
    targetTable.rename(columns={'HMSTROKE': 'his_CBSTROKE', 'HMHYPERT': 'his_HYPERTEN'}, inplace=True)
    targetTable = targetTable.loc[:, ['RID'] + variables]
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)

    merge_data = pd.merge(data, targetTable, on=['RID'], how='left')
    merge_data = merge_data.reset_index(drop=True)

    return merge_data.loc[:, base_columns]

def addColumns_MMSE(data):
    data.drop(columns=['MMSE'], inplace=True)
    base_columns = data.columns.values.tolist()
    columns = ['MMSE']
    old_variables = ['MMSCORE']
    base_columns += columns
    targetTable = pd.read_csv('./raw_data/ADNI_MMSE.csv', low_memory=False)
    targetTable.drop(columns=['VISCODE'], inplace=True)
    targetTable.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable.loc[targetTable['VISCODE'] == 'sc', 'VISCODE'] = 'bl'
    targetTable.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)
    targetTable = targetTable.loc[:, ['RID', 'VISCODE'] + old_variables]

    merge_data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')
    deep_merge_data = cp.deepcopy(merge_data)
    deep_merge_data = pd.concat([deep_merge_data, pd.DataFrame(columns=columns)], sort=False)

    for index, row in merge_data.iterrows():
        #line_MMSCORE
        line_MMSCORE = [row['MMSCORE']]
        if '' not in line_MMSCORE:
            if (row['MMSCORE'] == row['MMSCORE']):
                deep_merge_data.loc[index, 'MMSE'] = sum([int(a) for a in line_MMSCORE if int(a) not in [-1 , 0] ])

    return deep_merge_data.loc[:, base_columns]

def addColumns_TrailMaking(data):
    base_columns = data.columns.values.tolist()
    variables = ['trailA', 'trailB']
    base_columns += variables
    targetTable = './raw_data/ITEM.csv'
    targetTable = pd.read_csv(targetTable)
    targetTable.rename(columns={'TMT_PtA_Complete': 'trailA', 'TMT_PtB_Complete': 'trailB'}, inplace=True)
    targetTable = targetTable.loc[:, ['RID', 'VISCODE'] + variables]
    targetTable.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    targetTable.replace(-4, None, inplace=True)
    targetTable.replace(-1, None, inplace=True)

    merge_data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')
    deep_merge_data = cp.deepcopy(merge_data)
    deep_merge_data = pd.concat([deep_merge_data, pd.DataFrame(columns=variables)], sort=False)
    for index, row in merge_data.iterrows():
        if row['trailA'] != row['trailA']:
            row['trailA'] = 0
        if 0 <= int(row['trailA']) <= 150:
            deep_merge_data.loc[index, 'trailA'] = row['trailA']
        if row['trailB'] != row['trailB']:
            row['trailB'] = 0
        if 0 <= int(row['trailB']) <= 300:
            deep_merge_data.loc[index, 'trailB'] = row['trailB']

    deep_merge_data = deep_merge_data.loc[:, base_columns]
    return deep_merge_data.loc[:, base_columns]

def addColumns_dep(data):
    base_columns = data.columns.values.tolist()
    variables = ['his_DEPOTHR']
    base_columns += variables
    targetTable = pd.read_csv('./raw_data/ADNI_DXSUM_PDXCONV.csv')
    targetTable.drop(columns=['VISCODE'], inplace=True)
    targetTable.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable = targetTable.loc[:, ['RID', 'VISCODE', 'DXNODEP', 'Phase']]

    data = pd.merge(data, targetTable, on=['RID', 'VISCODE'], how='left')
    df = cp.deepcopy(data)

    for index, row in data.iterrows():
        if str(row['DXNODEP']) == '1.0':
            df.loc[index, 'his_DEPOTHR'] = '1'
        else:
            df.loc[index, 'his_DEPOTHR'] = '0'

    targetTable = pd.read_csv('./raw_data/BLCHANGE.csv')
    targetTable.drop(columns=['VISCODE'], inplace=True)
    targetTable.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
    targetTable = targetTable.loc[:, ['RID', 'VISCODE', 'BCDEPRES']]
    data = pd.merge(df, targetTable, on=['RID', 'VISCODE'], how='left')
    df = cp.deepcopy(data)

    for index, row in data.iterrows():
        if row['Phase'] != 'ADNI1':
            df.loc[index, 'his_DEPOTHR'] = row['BCDEPRES']

    return df.loc[:, base_columns]



if __name__ == '__main__':
    train_data = pd.read_csv('./raw_data/ADNIMERGE.csv', low_memory=False)
    train_data = addColumns_demograph(train_data)
    train_data = addColumns_diagnosis(train_data)
    train_data = addColumns_ADAS(train_data)
    train_data = addColumns_logicalmemory(train_data)
    train_data = addColumns_Moca(train_data)
    train_data = addColumns_NPIQ(train_data)
    train_data = addColumns_FAQ(train_data)
    train_data = addColumns_GDS(train_data)
    train_data = addColumns_medhist(train_data)
    train_data = addColumns_FHQ(train_data)
    train_data = addColumns_modhach(train_data)
    train_data = addColumns_MMSE(train_data)
    train_data = addColumns_TrailMaking(train_data)
    train_data = addColumns_dep(train_data)

    train_data['path'] = '../MRI_process/test_data/processed/npy/'

    train_data.rename(
        columns={'AGE': 'age', 'PTGENDER': 'gender', 'PTEDUCAT': 'education', 'MMSE': 'mmse'}, inplace=True)


    origin_data = pd.read_csv('./raw_data/MRI.csv')

    origin_data = origin_data.loc[:, ['RID', 'VISCODE', 'filename', 'SavePath']]
    train_data = pd.merge(train_data, origin_data, on=['RID', 'VISCODE'])
    train_data.drop_duplicates(subset=['RID', 'VISCODE'], inplace=True)
    features = [
        "RID", "VISCODE", "path", 'SavePath', "filename", 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD',
        'ALL', 'OTHER', "age", "gender", "education", "trailA", "trailB", "boston", "digitB", "digitBL", "digitF",
        "digitFL", "animal", "gds", "lm_imm", "lm_del", "mmse", "npiq_DEL", "npiq_HALL", "npiq_AGIT", "npiq_DEPD", "npiq_ANX",
        "npiq_ELAT", "npiq_APA", "npiq_DISN", "npiq_IRR", "npiq_MOT", "npiq_NITE", "npiq_APP", "faq_BILLS", "faq_TAXES",
        "faq_SHOPPING", "faq_GAMES", "faq_STOVE", "faq_MEALPREP", "faq_EVENTS", "faq_PAYATTN", "faq_REMDATES", "faq_TRAVEL",
        "his_NACCFAM", "his_CVHATT", "his_CBSTROKE","his_HYPERTEN","his_DEPOTHR", "his_PSYCDIS", "his_ALCOHOL","his_SMOKYRS", "his_PACKSPER"
        ]
    train_data = train_data.loc[:, features]
    train_data.to_csv('./data/ADNI.csv', index=0)

    mri_only = ["RID", "VISCODE", 'path', 'SavePath', 'filename', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER']
    train_data_mri = train_data.loc[:, mri_only]
    train_data_mri.to_csv('./data/ADNI_mri.csv', index=0)

    #benifit
    benifit_data = pd.read_csv('./raw_data/benifit_data.csv')
    benifit_data = benifit_data.loc[:, ["RID", "VISCODE", "benefit"]]
    benifit_data = pd.merge(train_data, benifit_data, on=['RID', 'VISCODE'], how='left')
    benifit_data.to_csv('./data/ADNI_benifit.csv', index=0)


