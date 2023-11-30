[TOC]

# 服务器配置

发行版本：CentOS Linux release 7.9.2009

磁盘容量：17TB

运行内存：256GB

GPU：`A100 80GB PCIe` * 2

CUDA版本：11.2

# 安装编程环境

安装Python 3.9.12，然后执行以下命令安装项目所需的Python库：

```bash
pip install -r requirements.txt
```

# 数据预处理

与数据预处理相关的所有文件都存放在目录data_preprocess中

## 下载安装FSL

1. 在FSL官网[https://fsl.fmrib.ox.ac.uk/fsldownloads_registration](https://fsl.fmrib.ox.ac.uk/fsldownloads_registration)注册下载脚本fslinstaller.py。

2. 安装FSL：
在Linux（Centos 7）终端中，进入python3虚拟环境，切换当前路径到包含 fslinstaller.py 文件的文件夹，然后使用 python 运行它；如果希望安装到 ~/fsl/（默认路径）中，那么当安装程序询问安装位置时，只需按回车键即可。
例如，如果想下载到“Downloads”文件夹：

```bash
cd ~/Downloads
python fslinstaller.py
```

3. 接下来会自动下载fsl包并自动安装(stage 1, stage 2), 过程很长。
4. 默认安装路径是在/usr/local/

5. 检验是否安装成功：在linux终端输入`flirt -version`可查看相关的FSL版本。

6. 其他安装问题：详见官网[https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux)

## MRI数据、人口统计学数据预处理

1. 进入目录“demo_character_process”
2. 运行`MRI_analysis.py`，找出满足3 T、t1加权MRI扫描标准的MRI数据，中间数据保存在'./raw_data/MRI.csv'
3. 运行`combine.py`，合并所有队列中的所有患者，并将结果输出到data/中
4. 从根目录进入目录“data_preprocess”，运行`data_transform.py`，进行数据清理、分割，处理好的数据保存到dataset/中

# 训练模型

需要训练的模型有三个：MRI模型，nonImg模型，Fusion模型

## 训练MRI模型

MRI模型使用MRI影像数据输入训练，模型的架构、学习率、训练轮数可以在task_config.json中进行配置，一般使用默认配置即可。

运行以下指令来训练MRI模型：

```bash
python train_mri_model.py
```

在模型训练的过程中，会保存阶段性训练的模型，保存目录为：`checkpoint_dir/<model_name>/`，其中*model_name*是模型的名称，可以在`train_mri_model.py`中更改，例如将模型名称设置为**MRI_model_20231124**

```python
def main():
    model = Multask_Wrapper(
        tasks=['COG'],
        device=0,
        main_config={
            'model_name': 'MRI_model_20231124',  # 在此处修改模型名称
            'csv_dir': os.path.realpath(os.path.join(root_path, 'data_preprocess/dataset'))
        },
        task_config=read_json('task_config.json'),
        seed=1000
    )
    model.train()
```

MRI模型需要的训练时间较长，大约需要10小时左右

## 训练nonImg模型

nonImg模型使用人口统计学特征来训练，运行以下指令开始训练：

```bash
python train_nonimg_model.py
```

同样地，nonImg模型每训练一轮就会保存一次模型，保存目录为：`checkpoint_dir/<model_name>`，其中*model_name*是模型的名称，可以在train_nonimg_model.py中更改，例如将模型名称设置为**nonimg_model_20231124**

```python
def main():
    # 初始化
    model_name = 'nonimg_model_20231124'  # 在此处修改模型名称
```

此模型需要的训练时间较短

## 训练Fusion模型

Fusion模型是一个结合MRI模型和nonImg模型的融合模型，Fusion模型使用人口统计学特征加上`COG_Score`来训练，`COG_Score`是MRI模型的预测结果，所以训练Fusion模型前一定要先训练MRI模型，否则会出错。

训练Fusion模型需要运行代码文件`train_fusion_model.py`，在运行之前，需要在此文件中配置MRI模型名称及Fusion模型名称，MRI模型名称必须是已训练的MRI模型的名称，Fusion模型名称可自定义：

```python
def main():
    mri_model_name = 'MRI_model_20231124'  # 在此处修改MRI模型名称
    fusion_model_name = 'Fusion_model_20231124'  # 在此处修改Fusion模型名称
```

然后运行以下代码开始训练Fusion模型：

```bash
python train_fusion_model.py
```

Fusion模型每训练一轮就会保存一次模型，保存目录为：checkpoint_dir/<model_name>，其中*model_name*是模型的名称，在以上例子中，模型名称为**Fusion_model_20231124**

此模型所需的运行时间较短

# 计算相关指标

为了评估模型的性能以及后续画图展示，需要计算指标：`sensitivity`、`specificity`、`accuracy`、`auc`、`ap`、`benefit`，按照如下步骤运行代码：

1. 使用MRI模型预测测试集

这一步需要使用训练好的MRI模型来预测整个测试集获得COG_Score，在代码`mri_predict_test_set.py`中指定模型名称：

```python
if __name__ == '__main__':
    main(
        model_name='MRI_model_20231124',  # 在此处修改模型名称
        test_set_path=os.path.join(root_path, 'data_preprocess/dataset/test.csv'),
        result_save_path=os.path.join(root_path, 'model_eval/eval_result/mri/scores.npy')
    )
```

模型的预测结果将会保存在：`model_eval/eval_result/mri/scores.npy`，预测结果是一个形状为(num_model, num_testset)的二维数组，行数为保存的模型数量，列数为测试集样本量

2. 使用nonImg模型预测测试集

# 画图展示


