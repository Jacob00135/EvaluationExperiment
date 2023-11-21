import pandas as pd
import numpy as np

#Check for bl blank data in VISCODE column in all visit records of RID. If the RID contains sc or scmri or m03 non-blank data,
# one of sc,scmri or m03 data will be used to complete bl visit data.
def combine_m03_and_sc_into_bl(df, isImage=False, Modality='MRI'):
    rID_set = df['RID'].drop_duplicates()
    # df = df.set_index(['RID', 'VISCODE2'], drop=False)
    for index, row in rID_set.items():
        rRID = row
        if isImage:
            sub_df = df.loc[(df['RID'] == rRID) & (df['Modality'] == Modality) & (
                    (df['VISCODE'] == 'm03') | (df['VISCODE'] == 'bl') | (df['VISCODE'] == 'sc') | (
                    df['VISCODE'] == 'scmri'))]
        else:
            sub_df = df.loc[(df['RID'] == rRID) & (
                    (df['VISCODE'] == 'm03') | (df['VISCODE'] == 'bl') | (df['VISCODE'] == 'sc') | (
                    df['VISCODE'] == 'scmri'))]

        #if (row == '4858' and isImage and Modality == 'MRI'):
        #    print('%%%%%%%%%%%%%%%%%%%%%%%%%')
        #    print(sub_df)
        # print('xxxxxxxxxxxxxxxxxx',row)
        # print('子模块图像数量：',sub_df.shape[0])
        if sub_df.shape[0] > 0:
            bl_flag = sub_df['VISCODE'].isin(['bl']).any()
            sc_flag = sub_df['VISCODE'].isin(['sc', 'scmri']).any()
            m03_flage = sub_df['VISCODE'].isin(['m03']).any()
            ##bl_flag=sub_df['RID'].isin(['bl']).any()
            ##sc_flag=sub_df['RID'].isin(['sc','scmri']).any()
            ##m03_flage=sub_df['RID'].isin(['m03']).any()

            if bl_flag and (sc_flag or m03_flage):  # bl 与其他两个阶段中的一个或两个 共存
                columns = sub_df.columns.values.tolist()
                mark = []
                for index_order, row in sub_df.iterrows():
                    visit = row['VISCODE']
                    if visit == 'bl':
                        for i in range(0, len(columns)):
                            tmp_value = row[columns[i]]
                            if pd.isnull(tmp_value) or tmp_value == '-4' or tmp_value == '\"\"' or tmp_value == '':
                                mark.append([index_order, columns[i]])

                if not isImage:#不修补图像的基本信息
                    for i in range(0, len(mark)):
                        index = mark[i][0]
                        columns_name = mark[i][1]
                        for index_order, row in sub_df.iterrows():
                            visit = row['VISCODE']
                            if visit != 'bl':
                                other_value = row[columns_name]
                                if not (pd.isnull(
                                        other_value) or other_value == '-4' or other_value == '\"\"' or other_value == ''):
                                    # print('修改前：',df.at[index, columns_name])
                                    df.at[index, columns_name] = other_value
                                    # print('修改后：',df.at[index, columns_name])
                                    break
            elif (not bl_flag) and (sc_flag and m03_flage):  # bl 不存在，sc和m03都存在
                columns = sub_df.columns.values.tolist()
                mark = []
                change_visitcoede = []
                for index_order, row in sub_df.iterrows():
                    visit = row['VISCODE']
                    if visit == 'sc' or visit == 'scmri':
                        change_visitcoede.append(index_order)
                        for i in range(0, len(columns)):
                            tmp_value = row[columns[i]]
                            if pd.isnull(tmp_value) or tmp_value == '-4' or tmp_value == '\"\"' or tmp_value == '':
                                mark.append([index_order, columns[i]])

                if not isImage:#不修补图像的基本信息
                    for i in range(0, len(mark)):
                        index = mark[i][0]
                        columns_name = mark[i][1]
                        for index_order, row in sub_df.iterrows():
                            visit = row['VISCODE']
                            if not (visit == 'sc' or visit == 'scmri'):
                                other_value = row[columns_name]
                                if not (pd.isnull(
                                        other_value) or other_value == '-4' or other_value == '\"\"' or other_value == ''):
                                    # print('修改前：',df.at[index, columns_name])
                                    df.at[index, columns_name] = other_value
                                    # print('修改后：',df.at[index, columns_name])
                                    break

                for index in change_visitcoede:
                    # print('修改前：',df.at[index_order, 'VISCODE2'])
                    df.at[index, 'VISCODE'] = 'bl'
                    # print('修改后：',df.at[index_order, 'VISCODE2'])
            elif (not (bl_flag or m03_flage)) and sc_flag:  # 只有 sc 阶段存在
                for index_order, row in sub_df.iterrows():
                    # print('修改前：',df.at[index_order, 'VISCODE2'])
                    df.at[index_order, 'VISCODE'] = 'bl'
                    # print('修改后：',df.at[index_order, 'VISCODE2'])
            elif (not (bl_flag or sc_flag)) and m03_flage:  # 只有m03阶段存在
                for index_order, row in sub_df.iterrows():
                    # print('修改前：',df.at[index_order, 'VISCODE2'])
                    df.at[index_order, 'VISCODE'] = 'bl'
                    # print('修改后：',df.at[index_order, 'VISCODE2'])
        # print('xxxxxxxxxxxxxxxxxx',row)
        # print('xxxxxxxxxxxxxxxxxx',isImage)
        # print('xxxxxxxxxxxxxxxxxx',Modality)
        #if ((rRID == '4858') and isImage and (Modality == 'MRI')):
        #    print('%%%%%%%%%%%%%%%%%%%%%%%%%')
        #    sdf = df.loc[(df['RID'] == rRID) & (df['Modality'] == Modality)]
        #    print(sdf)
    return df

#df 缺失值填充
def set_missing_value(df):
    #df.where(df.notnull(),'-4')
    df =df.fillna('-4')
    df =df.where(df !='', '-4')
    df =df.where(df != '\"\"', '-4')
    return df



ADNIMERGE = pd.read_csv('./raw_data/ADNIMERGE.csv')
image_df = pd.read_csv('./raw_data/image_information.csv')

image_df.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
image_df.replace(-4, None, inplace=True)
image_df.replace(-1, None, inplace=True)
image_df.replace('-4.0', None, inplace=True)
image_df.replace('-1.0', None, inplace=True)
image_df.replace('-4', None, inplace=True)
image_df.replace('-1', None, inplace=True)
image_df.replace('', None, inplace=True)

image_df_MRI = image_df.loc[image_df['Modality'] == 'MRI']
image_df_MRI = image_df_MRI.sort_values(by=['RID', 'VISCODE','Description'])
image_df_MRI = image_df_MRI.loc[image_df_MRI['Sequence'] == 1]
image_df_MRI = image_df_MRI.loc[image_df_MRI['Weighting'] == 'T1']
# image_df_MRI = image_df_MRI.loc[image_df_MRI['Field Strength'] == 1.5]
image_df_MRI.dropna(subset=['SavePath'], inplace=True)
image_df_MRI = image_df_MRI.drop_duplicates(subset=['RID', 'VISCODE'])


image_df_MRI = combine_m03_and_sc_into_bl(image_df_MRI,isImage=True,Modality='MRI')
image_df_MRI=set_missing_value(image_df_MRI)

#++++++++++++++++++++++++++++++++++++++++
image_df_MRI = image_df_MRI.loc[:, ['RID', 'VISCODE', 'Image ID', 'SavePath']]

MRI_image = pd.merge(ADNIMERGE, image_df_MRI, on=['RID', 'VISCODE'], how='left')
MRI_image.dropna(subset=['DX'], inplace=True)
MRI_image = MRI_image.loc[:, ['RID', 'VISCODE', 'DX', 'SavePath']]
MRI_image['filename']= MRI_image['SavePath'].map(lambda x: str(str(x).split('/')[-1]).replace('.nii', '.npy'))
MRI_image.dropna(subset=['SavePath'], inplace=True)
MRI_image = MRI_image.drop_duplicates(subset=['RID', 'VISCODE'])


df3 = pd.read_csv('./raw_data/nature_imginfo.csv')
df3 = df3.loc[:, ['RID', 'VISCODE', 'Phase']]
df3 = pd.merge(df3, MRI_image, on=['RID', 'VISCODE'])
df3.to_csv('./raw_data/MRI.csv', index=0)




