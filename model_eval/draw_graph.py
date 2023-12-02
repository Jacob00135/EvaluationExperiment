"""
绘图代码
"""
import os
import sys
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict
from PIL import Image

now_path = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(now_path, '..')))
from config import root_path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DrawScatter(object):

    def __init__(self, data, var_name):
        """
        画<指标-benefit>散点图的类，把三个模型的数值画到同一个图中，可以放大散点图指定位置

        :param data: dict。是如下形式的字典：
                     {
                         'MRI': MRI only模型测试数据，形状为(模型个数, 2)的numpy数组，数组
                                的第一列是变量数值，第二列是benefit,
                         'nonImg': nonImg模型测试数据，形状与MRI only模型的类似,
                         'Fusion': Fusion模型测试数据，形状与MRI only模型的类似
                     }
        :param var_name: str。变量的名字，只能在'sensitivity', 'specificity', 'accuracy', 'auc_nc',
                         'auc_mci', 'auc_de', 'ap_nc', 'ap_mci', 'ap_de'中取
        :returns: None
        """
        self.data = data
        self.var_name = var_name
        self.__magnify_x = None
        self.__magnify_y = None

        self.init_data()

    def init_data(self):
        """
        对输入的数据按变量进行排序

        :return: None
        """
        for model_name, data in self.data.items():
            var = data[:, 0]
            benefit = data[:, 1]
            argsort = var.argsort()
            var = var[argsort]
            benefit = benefit[argsort]
            self.data[model_name] = np.vstack((var, benefit)).T

    @staticmethod
    def filter_section(data, x=(0, 1), y=(0, 1), x_thre=0.01, y_thre=0.01):
        """
        将特定区域内的点剔除掉，使该区域的点不那么密集

        :param data: pd.DataFrame。要处理的数据
        :param x: tuple。默认(0, 1)。区域横坐标区间
        :param y: tuple。默认(0, 1)。区域纵坐标区间
        :param x_thre: float。默认0.01。|x_j - x_i| <= x_thre作为点密集的判断条件，
                              越高被剔除的点越多
        :param y_thre: float。默认0.01。|y_j - y_i| <= y_thre 作为点密集的判断条件，
                              越高被剔除的点越多
        :return: pd.DataFrame。处理完毕的数据
        """
        # 令横坐标最大和纵坐标最大的点不在可排除范围内
        best_x = data.iloc[data['var'].values.argmax(), 0]
        best_y = data.iloc[data['benefit'].values.argmax(), 1]

        # 搜索距离近的点删除
        i = 0
        while i < data.shape[0] - 1:
            if not (x[0] <= data.iloc[i, 0] <= x[1] and y[0] <= data.iloc[i, 1] <= y[1]):
                i = i + 1
                continue
            j = i + 1
            while j < data.shape[0] and x[0] <= data.iloc[j, 0] <= x[1] and\
                    abs(data.iloc[j, 0] - data.iloc[i, 0]) <= x_thre:
                if y[0] <= data.iloc[j, 1] <= y[1] and abs(data.iloc[j, 1] - data.iloc[i, 1]) <= y_thre and\
                        data.iloc[j, 0] != best_x and data.iloc[j, 1] != best_y:
                    data = data.drop(data.index[j], axis=0)
                else:
                    j = j + 1
            i = i + 1
        return data

    @staticmethod
    def generate_figure(magnify=False, figsize=7, dpi=200):
        """
        生成画布

        :param magnify: bool，默认为False。若为True则生成两个子图的画布；若为False则生成一个子图的画布
        :param figsize: int. 画布的高度，默认为7
        :param dpi: int. 画布的dpi，默认为200
        :return: tuple。表示(画布对象, 完整散点图子图对象)
        """
        if magnify:
            fig = plt.figure(figsize=(2 * figsize, figsize), dpi=dpi)
            full_ax = plt.subplot(1, 2, 1)
        else:
            fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
            full_ax = plt.subplot(1, 1, 1)
        return fig, full_ax

    def draw_full_scatter(self, ax=None, xlabel=None, adjust_padding=None):
        """
        绘制完整的散点图

        :param ax: 子图对象。默认为None
        :param xlabel: str. x轴的标签，默认为None，意为使用变量名作为x轴的标签
        :param adjust_padding: dict. 调整画布与图片边距的字典，例：{'left': 0.1, 'top': 0.9}。
                               默认为None，若为None则不调整边距
        :return: None
        """
        if xlabel is None:
            xlabel = self.var_name
        plt.scatter(
            self.data['MRI'][:, 0], self.data['MRI'][:, 1], marker='o', c='#6868ff',
            s=15, lw=0.2, ec='#555555', label='MRI', zorder=2
        )
        plt.scatter(
            self.data['nonImg'][:, 0], self.data['nonImg'][:, 1], marker='^', c='#ff8b26',
            s=25, lw=0.2, ec='#555555', label='nonImg', zorder=4
        )
        plt.scatter(
            self.data['Fusion'][:, 0], self.data['Fusion'][:, 1], marker='v', c='#ff0000',
            s=35, lw=0.2, ec='black', label='Fusion', zorder=6
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        plt.xticks(ticks, fontsize=16)
        plt.yticks(ticks, fontsize=16)
        plt.xlabel(xlabel, fontsize=28)
        plt.ylabel('benefit', fontsize=28)
        plt.legend(fontsize=16)
        plt.grid(True, c='#eeeeee', ls='--', zorder=0)
        if adjust_padding is not None:
            plt.subplots_adjust(**adjust_padding)
        if ax is not None:
            ax.set_aspect('equal', adjustable='box')

    @staticmethod
    def show(show=True, save_path=None):
        """
        显示、保存画布

        :param show: bool，默认为True。是否显示画布。
        :param save_path: str，默认为None。保存画布的路径，若为None，则不保存。
        :return: None
        """
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    def set_magnify_x(self, value):
        """
        设置magnify_x的值，表示放大的散点图的x轴范围
        
        :param value: tuple or str。若为tuple类型，则表示(min_x, max_x)。
                                    若为str类型，只能是'auto'，表示自动截取，截取规则如下：
                                    寻找散点中最大的x值max_x，令magnify_x=(max_x - 0.2, max_x)
        :return: None
        """
        if isinstance(value, tuple):
            self.__magnify_x = value
        elif value == 'auto':
            max_x = max(
                self.data['MRI'][:, 0].max(),
                self.data['nonImg'][:, 0].max(),
                self.data['Fusion'][:, 0].max()
            )
            self.__magnify_x = (max_x - 0.2, max_x)
        else:
            raise TypeError('value只支持元组、或值为"auto"的字符串')

    def get_magnify_x(self):
        """
        获得magnify_x的值

        :return: tuple。
        """
        if self.__magnify_x is None:
            raise ValueError('magnify_x没有初值，可以使用实例方法set_magnify_x设置其值')
        return self.__magnify_x

    def set_magnify_y(self, value):
        """
        设置magnify_y的值，表示放大的散点图的y轴范围

        :param value: tuple or str。若为tuple类型，则表示(min_y, max_y)。
                                    若为str类型，只能是'auto'，表示自动截取，截取规则如下：
                                    寻找散点中最大的y值max_y，magnify_y=(max_y - 0.2, max_y)
        :return: None
        """
        if isinstance(value, tuple):
            self.__magnify_y = value
        elif value == 'auto':
            max_y = max(
                self.data['MRI'][:, 1].max(),
                self.data['nonImg'][:, 1].max(),
                self.data['Fusion'][:, 1].max()
            )
            self.__magnify_y = (max_y - 0.2, max_y)
        else:
            raise TypeError('value值支持元组、或值为"auto"的字符串')

    def get_magnify_y(self):
        """
        获得magnify_y的值

        :return: tuple。
        """
        if self.__magnify_y is None:
            raise ValueError('magnify_y没有初值，可以使用实例方法set_magnify_y设置其值')
        return self.__magnify_y
    
    def filter_scatter(self, thre=0.01):
        """
        剔除整个散点图中的散点，使得图中的散点不那么密集

        :param thre: float，默认为0.01。判断散点是否密集的阈值，越高被剔除的点越多
        :return: None
        """
        for model_name, data in self.data.items():
            data = pd.DataFrame(data, columns=['var', 'benefit'])
            filter_data = self.filter_section(data, x_thre=thre, y_thre=thre)
            self.data[model_name] = filter_data.values

    def filter_magnify_scatter(self, thre=0.005):
        """
        仅剔除放大的图中的散点，使得图中的散点不那么密集


        :param thre: float. 默认为0.005。判断散点是否密集的阈值，越高被剔除的点越多
        :return: None
        """
        magnify_x = self.get_magnify_x()
        magnify_y = self.get_magnify_y()
        for model_name, data in self.data.items():
            data = pd.DataFrame(data, columns=['var', 'benefit'])
            filter_data = self.filter_section(data, x=magnify_x, y=magnify_y, x_thre=thre, y_thre=thre)
            self.data[model_name] = filter_data.values

    def filter_four_section_scatter(self, thre1=0.01, thre2=0.005):
        """
        将散点图分成4个区域：1个区域放大，3个区域不放大，使用不同的阈值剔除两种区域
        的散点，使散点不那么密集

        :param thre1: float，默认为0.01。判断3个不放大区域的散点是否密集的阈值，
                      越高被剔除的点越多
        :param thre2: float，默认为0.003。判断1个放大区域的散点是否密集的阈值，
                      越高被剔除的点越多
        :return: None
        """
        magnify_x = self.get_magnify_x()
        magnify_y = self.get_magnify_y()
        section = [
            [0, magnify_x[0], magnify_y[0], magnify_y[1], thre1],
            [magnify_x[0], magnify_x[1], magnify_y[0], magnify_y[1], thre2],
            [0, magnify_x[0], 0, magnify_y[0], thre1],
            [magnify_x[0], magnify_x[1], 0, magnify_y[0], thre1]
        ]
        for start_x, end_x, start_y, end_y, thre in section:
            for model_name, data in self.data.items():
                data = pd.DataFrame(data, columns=['var', 'benefit'])
                filter_data = self.filter_section(
                    data, (start_x, end_x), (start_y, end_y), thre, thre
                )
                self.data[model_name] = filter_data.values

    @staticmethod
    def boundary_check(v, delta=0.01):
        """
        在完整图中显示放大图的边框区域时，为了防止边框贴合完整图的边框，需要修正边框的坐标值
        :param v: float. 边框的一个坐标分量值
        :param delta: float. 修正时的变化值，默认为0.01
        :return: 修正后的分量值
        """
        if v <= 0:
            v = v + delta
        elif v >= 1:
            v = v - delta
        return v

    def draw_magnify_border(self):
        """
        在完整的散点图中绘制被放大的区域的边框

        :return: None
        """
        magnify_x = list(self.get_magnify_x())
        magnify_y = list(self.get_magnify_y())
        for i in range(len(magnify_x)):
            magnify_x[i] = self.boundary_check(magnify_x[i], 0.005)
            magnify_y[i] = self.boundary_check(magnify_y[i], 0.005)
        plt.plot(
            [magnify_x[0], magnify_x[0], magnify_x[1], magnify_x[1], magnify_x[0]],
            [magnify_y[0], magnify_y[1], magnify_y[1], magnify_y[0], magnify_y[0]],
            c='red', ls='--', lw=2, zorder=14
        )

    def draw_magnify_scatter(self, point_size=(150, 250, 350), ax=None, legend_loc='best', xlabel=None,
                             ticks_fontsize=16, label_fontsize=28, legend_fontsize=16, adjust_padding=None,
                             ticks_step=None):
        """
        绘制放大区域的散点图
        :param point_size: tuple. 三个模型的散点大小，默认为(150, 250, 350)
        :param ax: 子图对象。默认为None
        :param legend_loc: str. 指定图例的位置，默认值为'best'。例如：'left upper'
        :param xlabel: str. 图的x轴标签，默认为None，使用变量名。可使用LaTeX语法，如：$x^2$
        :param ticks_fontsize: int. 坐标轴刻度字号
        :param label_fontsize: int. 坐标轴标签字号
        :param legend_fontsize: int. 图例字号
        :param adjust_padding: dict. 调整画布与图片边距的字典，例：{'left': 0.1, 'top': 0.9}。
                               默认为None，若为None则不调整边距
        :param ticks_step: float. x轴和y轴的刻度步长
        :return: None
        """
        # 获取被放大的区域的坐标，并检查被放大的区域是否是正方形
        magnify_x = self.get_magnify_x()
        magnify_y = self.get_magnify_y()
        num_xticks = int(magnify_x[1] * 100) - int(magnify_x[0] * 100) + 1
        num_yticks = int(magnify_y[1] * 100) - int(magnify_y[0] * 100) + 1
        if num_xticks != num_yticks:
            print('警告: 非正方形的放大图: {} -- {:.2f} -- {:.2f}'.format(
                xlabel,
                magnify_x[1] - magnify_x[0],
                magnify_y[1] - magnify_y[0]
            ))

        # 画散点
        plt.scatter(
            self.data['MRI'][:, 0], self.data['MRI'][:, 1], marker='o',
            s=point_size[0], lw=1, c='#6868ff', ec='#555555', label='MRI', zorder=2
        )
        plt.scatter(
            self.data['nonImg'][:, 0], self.data['nonImg'][:, 1], marker='^',
            s=point_size[1], lw=1, c='#ff8b26', ec='#555555', label='nonImg', zorder=4
        )
        plt.scatter(
            self.data['Fusion'][:, 0], self.data['Fusion'][:, 1], marker='v',
            s=point_size[2], lw=1, c='#ff0000', ec='#555555', label='Fusion', zorder=6
        )
        plt.xlim(*magnify_x)
        plt.ylim(*magnify_y)

        # 画x轴和y轴的刻度等
        if ticks_step is not None:
            xticks = list(np.arange(magnify_x[0], magnify_x[1], ticks_step))
            yticks = list(np.arange(magnify_y[0], magnify_y[1], ticks_step))
            xticks.append(magnify_x[1])
            yticks.append(magnify_y[1])
        else:
            xticks = []
            for i in range(0, num_xticks, 3) if num_xticks < 15 else range(0, num_xticks + 1, 4):
                xtick = round(magnify_x[0] + 0.01 * i, 2)
                xticks.append(xtick)
            yticks = []
            for i in range(0, num_yticks, 3) if num_yticks < 15 else range(0, num_yticks + 1, 4):
                ytick = round(magnify_y[0] + 0.01 * i, 2)
                yticks.append(ytick)
        plt.xticks(xticks, fontsize=ticks_fontsize)
        plt.yticks(yticks, fontsize=ticks_fontsize)
        plt.xlabel(xlabel, fontsize=label_fontsize)
        plt.ylabel('benefit', fontsize=label_fontsize)
        plt.legend(loc=legend_loc, fontsize=legend_fontsize)
        plt.grid(True, c='#eeeeee', ls='--', zorder=0)
        if adjust_padding is not None:
            plt.subplots_adjust(**adjust_padding)
        if ax is not None:
            ax.set_aspect('equal', adjustable='box')

    def in_magnify(self, x, y):
        """
        判断一个散点是否在放大区域内

        :param x: float。要判断的散点的横坐标
        :param y: float。要判断的散点的纵坐标
        :return: bool。True表示在放大区域内，False表示不在放大区域内
        """
        magnify_x = self.get_magnify_x()
        magnify_y = self.get_magnify_y()
        return magnify_x[0] <= x <= magnify_x[1] and magnify_y[0] <= y <= magnify_y[1]

    def set_text(self, mark=True, performance_loc=None, benefit_loc=None):
        """
        在放大散点图中标记最优散点的坐标

        :param mark: bool. 是否显示文本. 默认为True
        :param performance_loc: tuple. 默认为None。最优performance点文本在坐标系中的坐标，若为None则使用点所在坐标
        :param benefit_loc: tuple. 默认为None。最优benefit点文本在坐标系中的坐标，若为None则使用点所在坐标
        :return: None
        """
        # 寻找最大值
        var_index = 0
        var_model_name = 'MRI'
        benefit_index = 0
        benefit_model_name = 'MRI'
        for model_name, data in self.data.items():
            max_0 = data[:, 0].argmax()
            max_1 = data[:, 1].argmax()
            now_max_0 = self.data[var_model_name][var_index, 0]
            now_max_1 = self.data[benefit_model_name][benefit_index, 1]
            if now_max_0 < data[max_0, 0] and self.in_magnify(data[max_0, 0], data[max_0, 1]):
                var_index = max_0
                var_model_name = model_name
            if now_max_1 < data[max_1, 1] and self.in_magnify(data[max_1, 0], data[max_1, 1]):
                benefit_index = max_1
                benefit_model_name = model_name

        # 显示文本
        best_var_point = (self.data[var_model_name][var_index, 0], self.data[var_model_name][var_index, 1])
        best_benefit_point = (
            self.data[benefit_model_name][benefit_index, 0],
            self.data[benefit_model_name][benefit_index, 1]
        )
        if mark:
            if performance_loc is None:
                performance_loc = best_var_point
            if benefit_loc is None:
                benefit_loc = best_benefit_point
            plt.annotate(
                '({:.4f},{:.4f})'.format(*best_var_point),
                xy=best_var_point,
                xytext=best_var_point if performance_loc is None else performance_loc,
                arrowprops={'facecolor': 'green', 'arrowstyle': 'fancy'},
                fontsize=30,
                fontfamily='Consolas',
                zorder=12
            )
            plt.annotate(
                '({:.4f},{:.4f})'.format(*best_benefit_point),
                xy=best_benefit_point,
                xytext=best_benefit_point if benefit_loc is None else benefit_loc,
                arrowprops={'facecolor': 'green', 'arrowstyle': 'fancy'},
                fontsize=30,
                fontfamily='Consolas',
                zorder=12
            )
            # plt.text(
            #     *performance_loc, s='({:.4f},{:.4f})'.format(*best_var_point),
            #     fontdict={'family': 'Consolas', 'size': 40}
            # )
            # plt.text(
            #     *benefit_loc, s='({:.4f},{:.4f})'.format(*best_benefit_point),
            #     fontdict={'family': 'Consolas', 'size': 40}
            # )


def draw_heatmap(data, title=None, show=True, save_path=None):
    """
    绘制一张热力图

    :param data: numpy数组。要求形状为(模型个数, 2)，第一列是性能指标数值（如sensitivity）
                 第二列是benefit
    :param title: str，默认为None。热力图的标题，若为None则不显示标题
    :param show: bool，默认为True。是否在绘制完毕后显示图像。
    :param save_path: str，默认为None。图的保存路径，若为None则不保存
    """
    # 对数据按指标进行排序
    var = data[:, 0]
    benefit = data[:, 1]
    argsort = var.argsort()
    var = var[argsort]
    benefit = benefit[argsort]
    data = np.vstack((var, benefit)).T

    # 若模型个数过多，则需要经过一次数据筛选
    if len(data) > 1000:
        data_df = pd.DataFrame(data, columns=['var', 'benefit'])
        data_df = DrawScatter.filter_section(data_df)
        data = data_df.values

    # 计算绘制热力图所需的数据
    num_model = len(data)
    hm = np.zeros((num_model, num_model), 'float32')
    for i in range(num_model - 1):
        for j in range(i + 1, num_model):
            delta_var = data[j, 0] - data[i, 0]
            if delta_var <= 0:
                delta_var = 0.001
            min_benefit = min(data[i, 1], data[j, 1])
            if min_benefit == 0:
                min_benefit = 0.001
            rate = (data[j, 1] - data[i, 1]) / min_benefit / delta_var
            if rate < -2:
                rate = -2
            elif rate > 2:
                rate = 2
            hm[i, j] = rate
            # hm[j, i] = -rate

    # 画图
    plt.figure(figsize=(7, 7), dpi=100)
    ax = plt.subplot(1, 1, 1)
    plt.pcolor(hm, cmap='RdBu')
    cb = plt.colorbar(fraction=0.043, pad=0.1)
    cb.ax.tick_params(labelsize=30)
    ax.yaxis.set_ticks_position('right')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ticks = np.linspace(0, num_model - 1, 5, dtype='int')
    plt.xticks(ticks, fontsize=20)
    plt.yticks(ticks, fontsize=20)
    plt.xlabel('number of the model', fontsize=30)
    plt.ylabel('number of the model', fontsize=30)
    if title is not None:
        plt.title(title, fontsize=40)
    plt.subplots_adjust(top=0.9, left=0.08, bottom=0.1, right=0.9)
    ax.set_aspect('equal', adjustable='box')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def draw_all_test_scatter():
    """
    根据测试集的性能指标(accuracy等)绘制散点图并保存
    此函数的设计不是复用的

    :return: None
    """

    # 确定路径、载入数据
    model_data = {
        'MRI': pd.read_csv(os.path.join(root_path, 'model_eval/eval_result/mri/result.csv')),
        'nonImg': pd.read_csv(os.path.join(root_path, 'model_eval/eval_result/nonImg/result.csv')),
        'Fusion': pd.read_csv(os.path.join(root_path, 'model_eval/eval_result/Fusion/result.csv'))
    }
    scatter_save_path = os.path.join(root_path, 'model_eval/eval_result/images/all_test_set/scatter')

    # 图的配置
    full_figure_config = OrderedDict({
        'sensitivity': ('$sensitivity$', (0.67, 0.87), (0.79, 0.99), 'upper left', (0.74, 0.96), (0.74, 0.96), 0.04),
        'specificity': ('$specificity$', (0.79, 0.94), (0.78, 0.93), 'upper left', (0.85, 0.895), (0.85, 0.925), 0.03),
        'accuracy': ('$accuracy$', (0.72, 0.9), (0.8, 0.98), 'upper left', (0.83, 0.96), (0.78, 0.92), 0.03),
        'auc_nc': ('$AUC_{NC}$', (0.84, 1), (0.8, 0.96), 'upper left', (0.952, 0.96), (0.921, 0.93), 0.04),
        'auc_mci': ('$AUC_{MCI}$', (0.77, 0.95), (0.78, 0.96), 'upper left', (0.885, 0.96), (0.845, 0.928), 0.03),
        'auc_de': ('$AUC_{DE}$', (0.84, 1.0), (0.78, 0.94), 'upper left', (0.907, 0.895), (0.91, 0.925), 0.04),
        'ap_nc': ('$AP_{NC}$', (0.78, 0.96), (0.78, 0.96), 'upper left', (0.87, 0.96), (0.84, 0.93), 0.03),
        'ap_mci': ('$AP_{MCI}$', (0.8, 0.98), (0.78, 0.96), 'upper left', (0.895, 0.925), (0.855, 0.945), 0.03),
        'ap_de': ('$AP_{DE}$', (0.77, 0.95), (0.78, 0.96), 'upper left', (0.875, 0.925), (0.815, 0.942), 0.03)
    })
    magnify_figure_config = OrderedDict({
        'sensitivity': ('$sensitivity$', (0.67, 0.87), (0.79, 0.99), 'upper left', (0.74, 0.96), (0.74, 0.96), 0.04),
        'specificity': ('$specificity$', (0.79, 0.94), (0.78, 0.93), 'upper left', (0.835, 0.89), (0.835, 0.915), 0.03),
        'accuracy': ('$accuracy$', (0.72, 0.9), (0.8, 0.98), 'upper left', (0.82, 0.96), (0.775, 0.92), 0.03),
        'auc_nc': ('$AUC_{NC}$', (0.84, 1), (0.8, 0.96), 'upper left', (0.93, 0.945), (0.89, 0.925), 0.04),
        'auc_mci': ('$AUC_{MCI}$', (0.77, 0.95), (0.78, 0.96), 'upper left', (0.875, 0.945), (0.82, 0.92), 0.03),
        'auc_de': ('$AUC_{DE}$', (0.84, 1.0), (0.78, 0.94), 'upper left', (0.89, 0.895), (0.90, 0.925), 0.04),
        'ap_nc': ('$AP_{NC}$', (0.78, 0.96), (0.78, 0.96), 'upper left', (0.88, 0.945), (0.83, 0.925), 0.03),
        'ap_mci': ('$AP_{MCI}$', (0.8, 0.98), (0.78, 0.96), 'upper left', (0.895, 0.925), (0.855, 0.945), 0.03),
        'ap_de': ('$AP_{DE}$', (0.77, 0.95), (0.78, 0.96), 'upper left', (0.875, 0.92), (0.82, 0.945), 0.03)
    })

    for var_name in magnify_figure_config.keys():
        # 准备数据
        scatter_data = {}
        for model_name, data in model_data.items():
            var = data[var_name].values
            benefit = data['benefit'].values
            scatter_data[model_name] = np.vstack((var, benefit)).T
        filter_thre = 0.008

        # region 画全图(包括两个子图，左边完整图，右边放大图)
        xlabel, magnify_x, magnify_y, legend_loc, perf_loc, benefit_loc, ticks_step = full_figure_config[var_name]
        ds = DrawScatter(deepcopy(scatter_data), var_name)
        plt.figure(figsize=(10, 4.6), dpi=300)
        left_ax = plt.subplot(1, 2, 1)
        ds.set_magnify_x(magnify_x)
        ds.set_magnify_y(magnify_y)
        ds.draw_full_scatter(ax=left_ax, xlabel=xlabel, adjust_padding={
            'left': 0.06, 'bottom': 0.17, 'top': 0.96})
        ds.draw_magnify_border()
        right_ax = plt.subplot(1, 2, 2)
        ds.filter_magnify_scatter(filter_thre)
        ds.draw_magnify_scatter(point_size=(150, 250, 350), ax=right_ax, legend_loc=legend_loc, xlabel=xlabel,
                                ticks_fontsize=16, label_fontsize=28, legend_fontsize=16,
                                ticks_step=ticks_step, adjust_padding={'right': 0.99, 'bottom': 0.17, 'top': 0.96})
        # ds.set_text(True, perf_loc, benefit_loc)
        ds.show(show=False, save_path=os.path.join(scatter_save_path, '{}_scatter.png'.format(var_name)))
        # endregion

        # region 画放大图(一个长宽比例接近2:1的放大图)
        xlabel, magnify_x, magnify_y, legend_loc, perf_loc, benefit_loc, ticks_step = magnify_figure_config[var_name]
        ds = DrawScatter(deepcopy(scatter_data), var_name)
        plt.figure(figsize=(10, 5), dpi=300)
        ds.set_magnify_x(magnify_x)
        ds.set_magnify_y(magnify_y)
        ds.filter_magnify_scatter(filter_thre)
        ds.draw_magnify_scatter(point_size=(250, 350, 450), legend_loc=legend_loc, xlabel=xlabel, ticks_step=ticks_step,
                                ticks_fontsize=24, label_fontsize=34, legend_fontsize=24,
                                adjust_padding={'left': 0.12, 'right': 0.96, 'bottom': 0.18, 'top': 0.96})
        ds.set_text(True, perf_loc, benefit_loc)
        # ds.show(show=False, save_path='C:/Users/330c-001/Desktop/tmp.png')
        ds.show(show=False, save_path=os.path.join(scatter_save_path, '{}_magnify_scatter.png'.format(var_name)))
        # endregion


def draw_all_test_heatmap():
    """
    根据测试集的性能指标(accuracy等)绘制热力图并保存
    此函数的设计不是复用的

    :return: None
    """
    model_data = {
        'MRI': pd.read_csv(os.path.join(root_path, 'model_eval/eval_result/mri/result.csv')),
        'nonImg': pd.read_csv(os.path.join(root_path, 'model_eval/eval_result/nonImg/result.csv')),
        'Fusion': pd.read_csv(os.path.join(root_path, 'model_eval/eval_result/Fusion/result.csv'))
    }
    var_name_list = {
        'sensitivity': ('$sensitivity$', ),
        'specificity': ('$specificity$', ),
        'accuracy': ('$accuracy$', ),
        'auc_nc': ('$AUC_{NC}$', ),
        'auc_mci': ('$AUC_{MCI}$', ),
        'auc_de': ('$AUC_{DE}$', ),
        'ap_nc': ('$AP_{NC}$', ),
        'ap_mci': ('$AP_{MCI}$', ),
        'ap_de': ('$AP_{DE}$', )
    }
    heatmap_path = os.path.join(root_path, 'model_eval/eval_result/images/all_test_set/heatmap')
    for var_name, (title, ) in var_name_list.items():
        for model_name, data in model_data.items():
            var = data[var_name].values
            benefit = data['benefit'].values
            heatmap_data = np.vstack((var, benefit)).T
            draw_heatmap(
                heatmap_data,
                title='{} for {}'.format(title, model_name),
                show=False,
                save_path=os.path.join(heatmap_path, '{}_{}_heatmap'.format(var_name, model_name))
            )


def coord_transform(value, max_value):
    """
    将参数坐标转换成真实坐标

    :param value: int or str. 参数坐标，形式可以是：100, -100, '100', '-100'
    :param max_value: int. 最大坐标(边界).
    :return: int
    """
    if isinstance(value, int):
        if value >= 0:
            return value
        return max_value + value
    if not isinstance(value, str):
        raise TypeError('value参数只能是int或str类型')
    if value[0] != '-':
        return int(value)
    return max_value - int(value[1:])


def crop_batch_image(src_path, dst_path, box, img_extension=('png', 'jpg')):
    """
    批量裁剪图片

    :param src_path: str. 原图片存放目录路径
    :param dst_path: str. 裁剪后图片存放目录路径，若不存在则新建
    :param box: 4-tuple. 图片裁剪的坐标(left, upper, right, lower)
    :param img_extension: tuple. 筛选图片的扩展名，只有扩展名在此变量中的图片才会被裁剪
    :return: None.
    """
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for fn in os.listdir(src_path):
        if fn.rsplit('.', 1)[-1] != img_extension:
            continue
        img = Image.open(os.path.join(src_path, fn))
        new_img = img.crop(box=(
            box[0],
            box[1],
            coord_transform(box[2], img.width),
            coord_transform(box[3], img.height)
        ))
        new_img.save(os.path.join(dst_path, fn))


if __name__ == '__main__':
    """
    在图中展示的散点，是经过了筛选的：
    由于散点太密集，所以在放大的散点图中，剔除掉了一些点以便观察
    剔除散点的规则如下：
    1.对于两个点(x_i, y_i), (x_j, y_j)，若|x_i - x_j| <= thre且|y_i - y_j| <= thre则剔除其中一点，thre取0.08
    2.对于三个模型的点，横坐标最大及纵坐标最大的点不会被剔除，因为这些点很重要
    """

    """
    三个模型的个数为何不一致(分别有1624、300、100个保存点)
    MRI是深度学习模型，训练时收敛慢，需要的训练步较多，所以能保存更多模型
    nonImg和Fusion都是CatBoost模型，训练时收敛很快，所以只能保存较少模型
    """

    """
    benefit的计算中，两次随访间隔在52周内
    """

    """
    检查散点图：
    1.箭头与文本的位置是否合适
    2.坐标轴刻度是否均匀、两端有刻度、合理
    3.图例位置是否遮挡散点
    4.画布的四个边是否贴近图片边缘
    """
    draw_all_test_scatter()
    draw_all_test_heatmap()
