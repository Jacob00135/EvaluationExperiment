import os
import numpy as np
import pandas as pd
from random import shuffle
from collections import Counter
from pandas import DataFrame

# region 路径相关变量
root_path = os.path.realpath(os.path.dirname(__file__))
data_path = os.path.realpath(os.path.join(root_path, 'our_benifit_data'))
nonimg_data_path = os.path.realpath(os.path.join(root_path, 'nonImg_data'))
# endregion

# region 数据相关变量
class_label_map = {
    0: 'NC',
    1: 'MCI',
    2: 'DE'
}
label_class_map = {
    'NC': 0,
    'MCI': 1,
    'DE': 2
}


# endregion


class Preprocess(object):
    dataset_filenames = []

    @staticmethod
    def read_data_from_datadir(filename: str, random_shuffle: bool = False) -> DataFrame:
        # 读取数据
        file_path = os.path.realpath(os.path.join(data_path, filename))
        if not os.path.exists(file_path):
            raise FileNotFoundError('找不到数据：{}'.format(file_path))
        df = pd.read_csv(file_path)

        # 打乱数据顺序
        if random_shuffle:
            df = Preprocess.shuffle(df)

        return df

    @staticmethod
    def shuffle(df: DataFrame) -> DataFrame:
        index = df.index.to_list()
        shuffle(index)
        df = df.loc[index, :]
        df.index = range(df.shape[0])
        return df

    @staticmethod
    def save_csv(df: DataFrame, filename: str) -> str:
        save_path = os.path.realpath(os.path.join(data_path, filename))
        df.to_csv(save_path, index=False)
        return save_path

    @staticmethod
    def split_filename(filename: str, split_char: str = '.') -> tuple:
        i = filename.rfind(split_char)
        if i < 0:
            return filename, ''
        return filename[:i], filename[i + 1:]

    @staticmethod
    def concat(dfs: iter) -> DataFrame:
        df = pd.concat(dfs, axis=0)
        df.index = range(df.shape[0])
        return df

    @property
    def shape(self) -> DataFrame:
        shape_map = {
            'dataset': [],
            'shape': []
        }
        for df, fn, full_fn in self.dataset_generator():
            shape_map['dataset'].append(fn)
            shape_map['shape'].append(df.shape)

        return pd.DataFrame(shape_map)

    def load_data(self, random_shuffle: bool = False) -> None:
        # 读取数据
        for full_filename in self.dataset_filenames:
            filename = full_filename.rsplit('.', 1)[0]
            df = self.read_data_from_datadir(full_filename, random_shuffle=random_shuffle)
            setattr(self, filename, df)

    def dataset_generator(self, dataset_filenames: list = None) -> iter:
        if dataset_filenames is None:
            dataset_filenames = self.dataset_filenames
        for full_fn in dataset_filenames:
            fn = full_fn.rsplit('.', 1)[0]
            yield getattr(self, fn), fn, full_fn


class RawDatasetPreprocess(Preprocess):
    dataset_filenames = [
        'our_cn_test.csv',
        'our_cn_train.csv',
        'our_Dementia_benifit_test.csv',
        'our_Dementia_benifit_train.csv',
        'our_Dementia_no_benifit_test.csv',
        'our_Dementia_no_benifit_train.csv',
        'our_MCI_benifit_test.csv',
        'our_MCI_benifit_train.csv',
        'our_MCI_no_benifit_test.csv',
        'our_MCI_no_benifit_train.csv'
    ]

    def __init__(self, load=True):
        super(RawDatasetPreprocess, self).__init__()

        self.our_cn_test = None
        self.our_cn_train = None
        self.our_Dementia_benifit_test = None
        self.our_Dementia_benifit_train = None
        self.our_Dementia_no_benifit_test = None
        self.our_Dementia_no_benifit_train = None
        self.our_MCI_benifit_test = None
        self.our_MCI_benifit_train = None
        self.our_MCI_no_benifit_test = None
        self.our_MCI_no_benifit_train = None

        if load:
            self.load_data()

    def concat_train_test(self, save: bool = False) -> None:
        # 分类
        df_dict = {}
        for df, fn, full_fn in self.dataset_generator():
            key = fn.rsplit('_', 1)[0]
            if key in df_dict:
                df_dict[key].append(df)
            else:
                df_dict[key] = [df]

        # 合并
        for var_name, dfs in df_dict.items():
            new_df = self.concat(dfs)
            setattr(self, var_name, new_df)
            if save:
                self.save_csv(new_df, '{}.csv'.format(var_name))

    @staticmethod
    def get_dataset_label(dataset_name: str) -> str:
        """获得数据集对应的标签"""
        prefix_map = {
            'our_cn': 'NC',
            'our_MCI': 'MCI',
            'our_De': 'DE'
        }
        for prefix, label in prefix_map.items():
            if dataset_name.startswith(prefix):
                return label

    def find_exceptional_label(self) -> None:
        # 找出异常样本
        except_df = pd.DataFrame(columns=['dataset'] + self.our_cn_test.columns.to_list())
        for df, fn, full_fn in self.dataset_generator():
            dataset_label = self.get_dataset_label(fn)
            except_index = list(filter(lambda i: df.loc[i, dataset_label] != 1, df.index))
            temp_df = df.loc[except_index, :].copy()
            temp_df.insert(0, 'dataset', full_fn)
            except_df = self.concat((except_df, temp_df))

        # 检查是否有异常
        if except_df.shape[0] <= 0:
            print('没有异常样本！')
            return None

        # 合并
        df1 = self.read_data_from_datadir('ADNIMERGE_without_bad_value.csv')
        df2 = self.read_data_from_datadir('ADNI_DXSUM_PDXCONV.csv')
        except_df = except_df.set_index(['RID', 'VISCODE'])
        df1 = df1.set_index(['RID', 'VISCODE']).loc[except_df.index, :]
        df2 = df2.set_index(['RID', 'VISCODE']).loc[except_df.index, :]

        # 保存
        self.save_csv(except_df, 'exceptional_sample.csv')
        self.save_csv(df1, 'ADNIMERGE_without_bad_value_exception.csv')
        self.save_csv(df2, 'ADNI_DXSUM_PDXCONV_exception.csv')


class ConcatDatasetPreprocess(Preprocess):
    dataset_filenames = [
        'our_cn.csv',
        'our_Dementia_benifit.csv',
        'our_Dementia_no_benifit.csv',
        'our_MCI_benifit.csv',
        'our_MCI_no_benifit.csv'
    ]

    def __init__(self, load: bool = True):
        super(ConcatDatasetPreprocess, self).__init__()

        self.our_cn = None
        self.our_Dementia_benifit = None
        self.our_Dementia_no_benifit = None
        self.our_MCI_benifit = None
        self.our_MCI_no_benifit = None

        if load:
            self.load_data(random_shuffle=True)

    def merge_adas13_column(self) -> None:
        merge_table = Preprocess.read_data_from_datadir('ADNIMERGE_without_bad_value.csv')[['RID', 'VISCODE', 'ADAS13']]
        for dataset, fn, full_fn in self.dataset_generator():
            new_dataset = pd.merge(dataset, merge_table, on=['RID', 'VISCODE'])
            new_dataset.index = range(new_dataset.shape[0])
            Preprocess.save_csv(new_dataset, '{}_adas13.csv'.format(fn))

    @staticmethod
    def compute_benefit(data: DataFrame) -> DataFrame:
        # 添加一列用于标识随访时间
        months = np.zeros(data.shape[0], int)
        for i, viscode in enumerate(data['VISCODE'].values):
            if viscode == 'bl':
                months[i] = 0
            else:
                months[i] = int(viscode[1:])
        data.insert(2, 'months', months)

        # 排序
        data = data.sort_values(['RID', 'months'])
        data.index = range(data.shape[0])

        # 计算benefit
        benefit = np.zeros(data.shape[0], 'float32')
        i = 0
        while i < data.shape[0]:
            rid = data.loc[i, 'RID']

            # 寻找第一个不是nan的ADAS13
            before_adas13 = data.loc[i, 'ADAS13']
            i = i + 1
            while pd.isna(before_adas13) and i < data.shape[0] and data.loc[i, 'RID'] == rid:
                benefit[i - 1] = np.nan
                before_adas13 = data.loc[i, 'ADAS13']
                i = i + 1

            # 没有找到不是nan的ADAS13
            if pd.isna(before_adas13):
                benefit[i - 1] = np.nan
                continue

            # 找到了第一个不是nan的ADAS13
            before_index = i - 1
            while i < data.shape[0] and data.loc[i, 'RID'] == rid:
                now_adas13 = data.loc[i, 'ADAS13']
                if pd.isna(now_adas13):
                    benefit[i] = np.nan
                else:
                    diff = before_adas13 - now_adas13
                    if diff > 0:
                        benefit[before_index] = diff
                    else:
                        benefit[before_index] = 0
                    before_adas13 = now_adas13
                    before_index = i
                i = i + 1
            benefit[before_index] = np.nan

        # 插入benefit字段
        data.insert(data.shape[1], 'benefit', benefit)
        data.index = range(data.shape[0])

        return data

    def set_benefit_column(self) -> None:
        self.our_cn.insert(self.our_cn.shape[1], 'benefit', np.nan)
        self.save_csv(self.our_cn, 'cn.csv')
        self.save_csv(
            df=self.compute_benefit(
                self.concat((self.our_MCI_benifit, self.our_MCI_no_benifit))
            ).drop(columns='months'),
            filename='mci.csv'
        )
        self.save_csv(
            df=self.compute_benefit(
                self.concat((self.our_Dementia_benifit, self.our_Dementia_no_benifit))
            ).drop(columns='months'),
            filename='de.csv'
        )


class FinalDatasetPreprocess(Preprocess):
    dataset_filenames = [
        'cn.csv',
        'mci.csv',
        'de.csv'
    ]

    def __init__(self, load: bool = True):
        super(FinalDatasetPreprocess, self).__init__()

        self.cn = None
        self.mci = None
        self.de = None

        if load:
            self.load_data(random_shuffle=True)

    def split(self, save: bool = False) -> tuple:
        # region 数据集划分方案
        """
        划分数据集要求
        ------------------------------------------------------------------------------------------------
        1. train_set : valid_set: test_set ≈ 6 : 2 : 2
        2. train_set = train_cn + train_de + train_mci
           valid_set = valid_cn + valid_de + valid_mci
           test_set = test_cn + test_de + test_mci
        3. train_cn : valid_cn : test_cn ≈ 6 : 2 : 2
           train_de : valid_de : test_de ≈ 6 : 2 : 2
           train_mci : valid_mci : test_mci ≈ 6 : 2 : 2
        4. test_de、test_mci中的条目要求：benefit字段值不为空、不为0
        ------------------------------------------------------------------------------------------------
        """
        # endregion

        # NC类别的划分
        n = round(self.cn.shape[0] * 0.2)
        test_cn = self.cn[:n]
        valid_cn = self.cn[n:2 * n]
        train_cn = self.cn[2 * n:]

        # MCI类别的划分
        n = round(self.mci.shape[0] * 0.3)
        test_mci_bool = np.zeros(self.mci.shape[0], 'bool')
        test_mci_index = 0
        for i, v in enumerate(self.mci['benefit'].values):
            if pd.notna(v) and v != 0:
                test_mci_bool[i] = True
                test_mci_index = test_mci_index + 1
                if test_mci_index >= n:
                    break
        test_mci = self.mci[test_mci_bool]
        n = round(self.mci.shape[0] * 0.2)
        valid_mci = self.mci[~test_mci_bool][:n]
        train_mci = self.mci[~test_mci_bool][n:]

        # DE类别的划分
        n = round(self.de.shape[0] * 0.2)
        test_de_bool = np.zeros(self.de.shape[0], 'bool')
        test_de_index = 0
        for i, v in enumerate(self.de['benefit'].values):
            if pd.notna(v) and v != 0:
                test_de_bool[i] = True
                test_de_index = test_de_index + 1
                if test_de_index >= n:
                    break
        test_de = self.de[test_de_bool]
        valid_de = self.de[~test_de_bool][:n]
        train_de = self.de[~test_de_bool][n:]

        # 合并三个类别
        train_set = self.concat((train_cn, train_de, train_mci))
        valid_set = self.concat((valid_cn, valid_de, valid_mci))
        test_set = self.concat((test_cn, test_de, test_mci))

        # 打乱数据次序后保存
        if save:
            self.save_csv(Preprocess.shuffle(train_set), 'train.csv')
            self.save_csv(Preprocess.shuffle(valid_set), 'valid.csv')
            self.save_csv(Preprocess.shuffle(test_set), 'test.csv')

        return train_set, valid_set, test_set


class Dataset(Preprocess):
    dataset_filenames = [
        'train.csv',
        'valid.csv',
        'test.csv'
    ]

    def __init__(self, load: bool = True):
        super(Dataset, self).__init__()

        self.train = None
        self.valid = None
        self.test = None

        if load:
            self.load_data(random_shuffle=True)


if __name__ == '__main__':
    # region MRI_only数据处理
    # FinalDatasetPreprocess().split(True)
    # d = Dataset()
    # print(d.train.shape)  # (3598, 85)
    # print(d.train.columns)  # nonImg_task_config.json中指定66个训练字段，所以此处有19个是非训练字段
    # for dataset, fn, full_fn in d.dataset_generator():
    #     benefit = dataset['benefit'].values
    #     nan_count = np.isnan(benefit).sum()
    #     print(fn, nan_count)
    # endregion

    # region nonImg数据探索
    """
    constant作空值填充：
    1. 得到的数据中，只有8个字段满足：字段中出现次数最多的值不超过样本总数的50%
    """
    # endregion
    # data = pd.read_csv(os.path.realpath(os.path.join(nonimg_data_path, 'nonImg_train_data_constant.csv')))
    #
    # for c in data.columns:
    #     value, count = Counter(data[c].values).most_common(1)[0]
    #     if count < data.shape[0] * 0.5:
    #         print('{}\t{:.2f}\t{:.2%}'.format(c, value, count / data.shape[0]))
    d = Dataset()
    data = d.concat((d.train, d.valid, d.test))
    for c in data.columns:
        na_percent = pd.isna(data[c]).sum() / data.shape[0]
        if na_percent > 0.1:
            print('{}\t\t{:.2%}'.format(c, na_percent))
