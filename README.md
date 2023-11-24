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
2. 运行`MRI_analysis.py`，找出满足3 T、t1加权MRI扫描标准的MRI数据
3. 运行`combine.py`，合并所有队列中的所有患者，并将结果输出到data/中
4. 从根目录进入目录“data_preprocess”，运行`data_transform.py`，进行数据清理、分割，处理好的数据保存到dataset/中

# 训练模型

# 计算相关指标

# 画图展示
