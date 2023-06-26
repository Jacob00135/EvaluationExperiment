"""
根据模型的预测结果，计算指标
"""
import os
import sys
import numpy as np
import pandas as pd

now_path = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(now_path, '..')))
from config import root_path, category_list


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
    main(
        scores_path=os.path.join(root_path, 'model_eval/eval_result/Fusion/scores.npy'),
        test_set_path=os.path.join(root_path, 'lookupcsv/CrossValid/no_cross/test_source.csv'),
        result_save_path=os.path.join(root_path, 'model_eval/eval_result/Fusion/result.csv')
    )
