"""
根据模型的预测结果，计算指标
"""
import os
import sys
import pdb
import warnings
import numpy as np
import pandas as pd
from time import time as get_timestamp
from scipy.stats import shapiro

now_path = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(now_path, '..')))
from config import root_path, category_list, category_map


class ConfusionMatrix(object):

    def __init__(self, real, pred):
        self.tp, self.fn, self.fp, self.tn = 0, 0, 0, 0
        for r, p in zip(real, pred):
            if r == 1 and r == p:
                self.tp = self.tp + 1
            elif r == 1:
                self.fn = self.fn + 1
            elif r == p:
                self.tn = self.tn + 1
            else:
                self.fp = self.fp + 1

    def get_fpr(self):
        return self.fp / (self.fp + self.tn)

    def get_tpr(self):
        return self.tp / (self.tp + self.fn)

    def get_sensitivity(self):
        return self.tp / (self.tp + self.fn)

    def get_specificity(self):
        return self.tn / (self.tn + self.fp)

    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        return self.tp / (self.tp + self.fn)

    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fn + self.tn + self.fp)


def compute_sensitivity(labels, score):
    sensitivity_sum = 0
    for i, category in enumerate(category_list):
        cm = ConfusionMatrix(labels[category], np.abs(score - i) < 0.5)
        sensitivity_sum = sensitivity_sum + cm.get_sensitivity()
    return sensitivity_sum / len(category_list)


def compute_specificity(labels, score):
    specificity_sum = 0
    for i, category in enumerate(category_list):
        cm = ConfusionMatrix(labels[category], np.abs(score - i) < 0.5)
        specificity_sum = specificity_sum + cm.get_specificity()
    return specificity_sum / len(category_list)


def compute_accuracy(label, score):
    return sum(label == score_to_prediction(score)) / len(label)


def compute_ADAS_benefit(data):
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


def compute_benefit(label, score, benefit):
    return sum(benefit[label == score_to_prediction(score)]) / sum(benefit)


def compute_auc(label, score, num_threshold=100):
    fprs = np.zeros(num_threshold, 'float32')
    tprs = np.zeros(num_threshold, 'float32')
    thresholds = np.linspace(min(score), max(score), num_threshold)
    for i, thre in enumerate(thresholds):
        cm = ConfusionMatrix(label, score <= thre)
        fprs[i], tprs[i] = cm.get_fpr(), cm.get_tpr()
    auc = 0
    for i in range(len(fprs) - 1):
        auc = auc + (tprs[i] + tprs[i + 1]) * (fprs[i + 1] - fprs[i]) / 2
    return auc


def compute_ap(label, score, num_threshold=100):
    recalls = np.zeros(num_threshold, 'float32')
    precisions = np.zeros(num_threshold, 'float32')
    thresholds = np.linspace(min(score), max(score), num_threshold)
    for i, thre in enumerate(thresholds):
        cm = ConfusionMatrix(label, score <= thre)
        recalls[i], precisions[i] = cm.get_recall(), cm.get_precision()
    ap = 0
    for i in range(len(recalls) - 1):
        ap = ap + (precisions[i] + precisions[i + 1]) * (recalls[i + 1] - recalls[i]) / 2
    return ap


def score_to_prediction(scores, thresholds=(0.5, 1.5)):
    prediction = np.zeros(len(scores), 'int')
    for i, score in enumerate(scores):
        if score > thresholds[1]:
            prediction[i] = 2
        elif score >= thresholds[0]:
            prediction[i] = 1
    return prediction


def compute(test_set, scores):
    # 初始化数据
    benefit = test_set['benefit'].fillna(0).values
    labels = {c: test_set[c].values for c in category_list}
    cog = test_set['COG'].values

    # 计算指标
    result = {
        'sensitivity': [],
        'specificity': [],
        'accuracy': [],
        'auc_nc': [],
        'auc_mci': [],
        'auc_de': [],
        'ap_nc': [],
        'ap_mci': [],
        'ap_de': [],
        'benefit': []
    }
    for i in range(len(scores)):
        score = scores[i, :]
        result['sensitivity'].append(compute_sensitivity(labels, score))
        result['specificity'].append(compute_specificity(labels, score))
        result['accuracy'].append(compute_accuracy(cog, score))
        result['benefit'].append(compute_benefit(cog, score, benefit))
        for j, category in enumerate(category_list):
            result['auc_{}'.format(category.lower())].append(compute_auc(labels[category], np.abs(score - j)))
            result['ap_{}'.format(category.lower())].append(compute_ap(labels[category], np.abs(score - j)))

    return pd.DataFrame(result)


class ComputeIndicator(object):

    def __init__(self, indicator_name, test_set, score):
        if indicator_name == 'benefit':
            self.indicator_name = 'benefit'
            self.function_params = {
                'label': test_set['COG'].values,
                'score': score,
                'benefit': test_set['benefit'].fillna(0).values
            }
        else:
            self.indicator_name, self.task = indicator_name.split('_', 1)
            self.task_index = category_map[self.task.upper()]
            self.function_params = {
                'label': test_set[self.task.upper()].values,
                'score': np.abs(score - self.task_index)
            }
        self.compute_function = eval('compute_{}'.format(self.indicator_name))

    def compute(self, index):
        params = {}
        for param_name, var in self.function_params.items():
            params[param_name] = var[index]
        return self.compute_function(**params)


def normal_3sigma_method(seq):
    """使用正态分布的3sigma原则计算置信区间"""
    # 对指标序列进行正态性检验
    statistic, p = shapiro(seq)
    is_normal = p > 0.05

    # 计算置信区间
    mean = seq.mean()
    std = seq.std()
    half = 1.96 * std / np.sqrt(len(seq))
    return mean - half, mean + half, is_normal


def fractile_method(seq):
    """使用百分位数计算置信区间"""
    seq = np.sort(seq)
    num_samping = len(seq)
    lower_i = max(int(np.floor(num_samping * 0.025)) - 1, 0)
    upper_i = min(int(np.ceil(num_samping * 0.975)) - 1, num_samping - 1)
    return seq[lower_i], seq[upper_i], True


def compute_ci(num_samping=100, save=True, method='3sigma'):
    """
    1.寻找7个模型：AUC_{NC}、AUC_{MCI}、AUC_{DE}、AP_{NC}、AP_{MCI}、AP_{DE}、benefit最好的模型，
      记为m1、m2、m3、m4、m5、m6、m7，这7个指标记为i1、i2、i3、i4、i5、i6、i7
    2.对于每一个模型，都进行如下步骤：
      (1)对测试集进行有放回抽取100次，得到100个数据集
      (2)对于100个数据集中的每一个数据集，都计算指标：m1只计算i1和i7、m2只计算i2和i7、m3只计算i3和i7、m4只计算i4和i7、
         m5只计算i5和i7、m6只计算i6和i7，m7需要计算i1~i7
      (3)经过步骤(2)后，每个指标会有100个值（例如对于i7，每个数据集可计算得到1个值，那么100个数据集就可以得到100个i7）
      (4)一个指标序列可以计算一个置信度为95%的置信区间，计算方式有两种：
         (i)使用正态分布的3sigma原则：(x - 1.96 * s / sqrt(100), x + 1.96 * s / sqrt(100))，其中x为平均值，s为标准差
         (ii)使用百分位数：先对100个指标序列按从小到大排序，则置信区间为：(2.5%分位数, 97.5%分位数)
    3.经过以上两步，可以得到19个置信区间：m1~m6各计算2个，m7计算得到7个
    """
    # 确定计算方式
    if method == '3sigma':
        compute_ci_function = normal_3sigma_method
    elif method == 'fractile':
        compute_ci_function = fractile_method
    else:
        raise ValueError('method参数只能取`3sigma`和`fractile`！')

    # 读取数据
    indicator_path = {
        'mri': os.path.join(root_path, 'model_eval/eval_result/mri/result.csv'),
        'nonImg': os.path.join(root_path, 'model_eval/eval_result/nonImg/result.csv'),
        'Fusion': os.path.join(root_path, 'model_eval/eval_result/Fusion/result.csv')
    }
    indicator = {}
    for model_name, p in indicator_path.items():
        indicator[model_name] = pd.read_csv(p)

    # 寻找MRI、nonImg、Fusion保存的模型中，7个指标最好的模型
    indicator_name_list = ['auc_nc', 'auc_mci', 'auc_de', 'ap_nc', 'ap_mci', 'ap_de', 'benefit']
    best_model = {}
    for model_name, data in indicator.items():
        for indicator_name in indicator_name_list:
            var = data[indicator_name].values
            index = var.argmax()
            value = var[index]
            if indicator_name not in best_model or (indicator_name in best_model and best_model[indicator_name][2] < value):
                best_model[indicator_name] = (model_name, index, value)

    # 读取原测试集
    test_set_path = os.path.join(root_path, 'lookupcsv/CrossValid/no_cross/test_source.csv')
    test_set = pd.read_csv(test_set_path)

    # 定义每个模型需要计算的指标
    compute_indicator = {
        'auc_nc': ['auc_nc', 'benefit'],
        'auc_mci': ['auc_mci', 'benefit'],
        'auc_de': ['auc_de', 'benefit'],
        'ap_nc': ['ap_nc', 'benefit'],
        'ap_mci': ['ap_mci', 'benefit'],
        'ap_de': ['ap_de', 'benefit'],
        'benefit': indicator_name_list
    }

    # 计算置信度95%的置信区间
    result = {}
    for indicator_name, (model_name, model_index, indicator_value) in best_model.items():
        start_time = get_timestamp()
        result[indicator_name] = []

        # 读取模型预测结果
        scores_path = os.path.join(root_path, 'model_eval/eval_result/{}/scores.npy'.format(model_name))
        scores = np.load(scores_path)[model_index]
        num_sample = len(scores)

        # 初始化计算指标的对象
        compute_obj_list = []
        for n in compute_indicator[indicator_name]:
            compute_obj = ComputeIndicator(n, test_set, scores)
            compute_obj_list.append(compute_obj)

        # 抽样并计算每一次抽样的指标
        samping_indicator = np.zeros((num_samping, len(compute_obj_list)), dtype='float32')
        for i in range(num_samping):
            random_index = np.random.randint(0, num_sample, num_sample)
            for j, compute_obj in enumerate(compute_obj_list):
                samping_indicator[i, j] = compute_obj.compute(random_index)

        # 计算置信度为95%的置信区间
        for j in range(samping_indicator.shape[1]):
            ci_lower, ci_upper, is_normal = compute_ci_function(samping_indicator[:, j])
            if not is_normal:
                warnings.warn('警告：正态性检验为假，但仍然使用3sigma原则计算CI。\n'
                              'model_name={} -- best_model={} -- indicator_name={}'.format(
                    model_name, indicator_name, compute_indicator[indicator_name][j]
                ))
            result[indicator_name].append((ci_lower, ci_upper))

        # 输出计时
        print('{} - {:.0f}s'.format(indicator_name, get_timestamp() - start_time))

    # 转换成表格并导出成excel
    df = {}
    name_list = ['auc_nc', 'auc_mci', 'auc_de', 'ap_nc', 'ap_mci', 'ap_de']
    for i, name in enumerate(name_list):
        df[name] = [
            '({:.4f}, {:.4f})'.format(*result[name][0]),
            '({:.4f}, {:.4f})'.format(*result[name][1]),
            '({:.4f}, {:.4f})'.format(*result['benefit'][i]),
            '({:.4f}, {:.4f})'.format(*result['benefit'][6])
        ]
    df = pd.DataFrame(df, index=['best_per_x', 'best_per_y', 'best_ben_x', 'best_ben_y'])
    if save:
        df.to_excel(os.path.join(root_path, 'model_eval/eval_result/ci.xlsx'))

    return df


def check_ci(ci):
    # 读取数据
    indicator_path = {
        'mri': os.path.join(root_path, 'model_eval/eval_result/mri/result.csv'),
        'nonImg': os.path.join(root_path, 'model_eval/eval_result/nonImg/result.csv'),
        'Fusion': os.path.join(root_path, 'model_eval/eval_result/Fusion/result.csv')
    }
    data = pd.DataFrame()
    indicator_name_list = ['auc_nc', 'auc_mci', 'auc_de', 'ap_nc', 'ap_mci', 'ap_de']
    for model_name, p in indicator_path.items():
        data = pd.concat((data, pd.read_csv(p)[indicator_name_list + ['benefit']]))
    data.index = range(data.shape[0])

    # 寻找MRI、nonImg、Fusion保存的模型中，最好的12个点
    index = ['best_per_x', 'best_per_y', 'best_ben_x', 'best_ben_y']
    best_point = {}
    best_benefit_index = data['benefit'].values.argmax()
    for indicator_name in indicator_name_list:
        i = data[indicator_name].values.argmax()
        best_point[indicator_name] = [
            data.loc[i, indicator_name],
            data.loc[i, 'benefit'],
            data.loc[best_benefit_index, indicator_name],
            data.loc[best_benefit_index, 'benefit']
        ]
    best_point = pd.DataFrame(best_point, index=index)

    # 值校验
    merge_array = np.zeros(best_point.shape, dtype='bool').tolist()
    check_array = np.zeros(best_point.shape, dtype='bool')
    reduce_merge_array = np.zeros((2, best_point.shape[1]), dtype='bool').tolist()
    reduce_check_array = np.zeros((2, best_point.shape[1]), dtype='bool')
    for c, indicator_name in enumerate(indicator_name_list):
        for r, i in enumerate(index):
            v = best_point.loc[i, indicator_name]
            ci_lower, ci_upper = ci.loc[i, indicator_name][1:-1].split(', ')
            ci_lower = float(ci_lower)
            ci_upper = float(ci_upper)
            merge_array[r][c] = '{:.4f} [CI: {:.4f}, {:.4f}]'.format(v, ci_lower, ci_upper)
            check_array[r, c] = ci_lower <= v <= ci_upper

        for r, lower_i, upper_i in zip([0, 1], ['best_per_x', 'best_ben_y'], ['best_ben_x', 'best_per_y']):
            v1 = round(best_point.loc[lower_i, indicator_name], 4)
            v2 = round(best_point.loc[upper_i, indicator_name], 4)
            ci_lower_1, ci_upper_1 = ci.loc[lower_i, indicator_name][1:-1].split(', ')
            ci_lower_1, ci_upper_1 = round(float(ci_lower_1), 4), round(float(ci_upper_1), 4)
            ci_lower_2, ci_upper_2 = ci.loc[upper_i, indicator_name][1:-1].split(', ')
            ci_lower_2, ci_upper_2 = round(float(ci_lower_2), 4), round(float(ci_upper_2), 4)
            v = v1 - v2
            ci_lower = ci_lower_1 - ci_upper_2
            ci_upper = ci_upper_1 - ci_lower_2
            reduce_merge_array[r][c] = '{:.4f}={:.4f}-{:.4f} [CI: {:.4f}, {:.4f}]'.format(v, v1, v2, ci_lower, ci_upper)
            reduce_check_array[r, c] = ci_lower <= v <= ci_upper

    merge_array = pd.DataFrame(merge_array, columns=indicator_name_list, index=index)
    check_array = pd.DataFrame(check_array, columns=indicator_name_list, index=index, dtype='bool')
    reduce_merge_array = pd.DataFrame(reduce_merge_array, columns=indicator_name_list, index=['performance', 'benefit'])
    reduce_check_array = pd.DataFrame(reduce_check_array, columns=indicator_name_list, index=['performance', 'benefit'], dtype='bool')
    return merge_array, check_array, reduce_merge_array, reduce_check_array


def main(scores_path, test_set_path, result_save_path):
    result = compute(pd.read_csv(test_set_path), np.load(scores_path))
    result.to_csv(result_save_path, index=False)


if __name__ == '__main__':
    # mri
    """
    main(
        scores_path=os.path.join(root_path, 'model_eval/eval_result/mri/scores.npy'),
        test_set_path=os.path.join(root_path, 'lookupcsv/CrossValid/no_cross/test_source.csv'),
        result_save_path=os.path.join(root_path, 'model_eval/eval_result/mri/result.csv')
    )
    """

    # nonImg
    """
    main(
        scores_path=os.path.join(root_path, 'model_eval/eval_result/nonImg/scores.npy'),
        test_set_path=os.path.join(root_path, 'lookupcsv/CrossValid/no_cross/test_source.csv'),
        result_save_path=os.path.join(root_path, 'model_eval/eval_result/nonImg/result.csv')
    )
    """

    # Fusion
    """
    main(
        scores_path=os.path.join(root_path, 'model_eval/eval_result/Fusion/scores.npy'),
        test_set_path=os.path.join(root_path, 'lookupcsv/CrossValid/no_cross/test_source.csv'),
        result_save_path=os.path.join(root_path, 'model_eval/eval_result/Fusion/result.csv')
    )
    """

    # 计算置信区间
    start_time = get_timestamp()
    i = 0
    while True:
        ci = compute_ci(num_samping=100, save=False, method='fractile')
        ci.to_excel('C:/Users/330c-001/Desktop/tmp.xlsx')
        # ci = pd.read_excel(os.path.join(root_path, 'model_eval/eval_result/ci.xlsx'), index_col=0)
        merge_array, check_array, reduce_merge_array, reduce_check_array = check_ci(ci)
        if np.sum(~check_array.to_numpy()) == 0 and np.sum(~reduce_check_array.to_numpy()) == 0:
            break
        i = i + 1
        print('已完成{}次CI计算与检验：已用时{:.0f}s\n'.format(i, get_timestamp() - start_time))
    print('完毕，总次数：{} -- 总用时：{:.0f}s'.format(i + 1, get_timestamp() - start_time))
