"""
根据模型的预测结果，计算指标
"""
import os
import sys
import json
import numpy as np
import pandas as pd

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


def compute_ci(num_samping=100):
    """
    1.寻找7个模型：AUC_{NC}、AUC_{MCI}、AUC_{DE}、AP_{NC}、AP_{MCI}、AP_{DE}、benefit最好的模型，
      记为m1、m2、m3、m4、m5、m6、m7，这7个指标记为i1、i2、i3、i4、i5、i6、i7
    2.对于每一个模型，都进行如下步骤：
      (1)对测试集进行有放回抽取100次，得到100个数据集
      (2)对于100个数据集中的每一个数据集，都计算指标：m1只计算i1和i7、m2只计算i2和i7、m3只计算i3和i7、m4只计算i4和i7、
         m5只计算i5和i7、m6只计算i6和i7，m7需要计算i1~i7
      (3)经过步骤(2)后，每个指标会有100个值（例如对于i7，每个数据集可计算得到1个值，那么100个数据集就可以得到100个i7），
         对这100个指标序列，求平均值x和标准差s
      (4)每一个指标都可以计算一个置信度为95%的置信区间：(x - 1.96 * s / sqrt(100), x + 1.96 * s / sqrt(100))
    3.经过以上两步，可以得到19个置信区间：m1~m6各计算2个，m7计算得到7个
    :param num_samping: int. 抽样的次数，默认为100
    :return: None
    """
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
    benefit = test_set['benefit'].fillna(0).values

    # 定义计算指标的函数
    def compute_auc_nc(index, score):
        return compute_auc(test_set['NC'].values[index], score)

    def compute_auc_mci(index, score):
        return compute_auc(test_set['MCI'].values[index], score)

    def compute_auc_de(index, score):
        return compute_auc(test_set['DE'].values[index], score)

    def compute_ap_nc(index, score):
        return compute_ap(test_set['NC'].values[index], score)

    def compute_ap_mci(index, score):
        return compute_ap(test_set['MCI'].values[index], score)

    def compute_ap_de(index, score):
        return compute_ap(test_set['DE'].values[index], score)

    def compute_benefit_2(index, score):
        return compute_benefit(test_set['COG'].values[index], score, benefit[index])

    # 定义每个模型需要计算的指标
    compute_function = {
        'auc_nc': [
            compute_auc_nc,
            compute_benefit_2
        ],
        'auc_mci': [
            compute_auc_mci,
            compute_benefit_2
        ],
        'auc_de': [
            compute_auc_de,
            compute_benefit_2
        ],
        'ap_nc': [
            compute_ap_nc,
            compute_benefit_2
        ],
        'ap_mci': [
            compute_ap_mci,
            compute_benefit_2
        ],
        'ap_de': [
            compute_ap_de,
            compute_benefit_2
        ],
        'benefit': [
            compute_auc_nc,
            compute_auc_mci,
            compute_auc_de,
            compute_ap_nc,
            compute_ap_mci,
            compute_ap_de,
            compute_benefit_2
        ]
    }

    # 计算置信度95%的置信区间
    result = {}
    for indicator_name, (model_name, model_index, indicator_value) in best_model.items():
        result[indicator_name] = []
        print(indicator_name, model_name, model_index, indicator_value)

        # 读取模型预测结果
        scores_path = os.path.join(root_path, 'model_eval/eval_result/{}/scores.npy'.format(model_name))
        scores = np.load(scores_path)[model_index]
        num_sample = len(scores)

        # 抽样并计算每一次抽样的指标
        func_list = compute_function[indicator_name]
        samping_indicator = np.zeros((num_samping, len(func_list)), dtype='float32')
        for i in range(num_samping):
            random_index = np.random.randint(0, num_sample, num_sample)
            score = scores[random_index]
            for j, f in enumerate(func_list):
                samping_indicator[i, j] = f(random_index, score)

        # 计算置信度为95%的置信区间
        for j in range(samping_indicator.shape[1]):
            var: np.ndarray = samping_indicator[:, j]
            mean = var.mean()
            std = var.std()
            half = 1.96 * std / np.sqrt(num_samping)
            section = (float(mean - half), float(mean + half))
            result[indicator_name].append(section)

    return result


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
    from time import time as get_timestamp
    start_time = get_timestamp()
    ci = compute_ci()
    print('用时：{:.0f}'.format(get_timestamp() - start_time))
    print(json.dumps(ci, indent=4, ensure_ascii=False))