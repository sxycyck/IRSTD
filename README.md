# 配置环境

## 所用环境版本

cuda 11.6

python 3.8

cudnn 8.9

paddlepaddle-gpu 2.4.2

paddle2onnx 1.0.5

显卡配置 NVIDIA A40,显存48G。

Python环境依赖文件路径为（paddlepaddle-gpu根据CUDA版本单独安装）：

```
project/requirements.txt
```

# 解决方案

## 数据增强

为了让模型能够更好的提取舰船特征，我们对原始数据集进行了数据增强策略。数据增强的运行代码路径为：

```
project/code/data_prepare/data_aug.py
```

运行脚本路径为：

```
project/code/data_aug.sh
```

增强后的数据集路径为：

```
project/user_data/aug_dataset
```

执行具体数据增强方法如下：

### 翻转及旋转

我们对原始数据集中的每一张图像进行随机水平翻转，垂直翻转或旋转，增加目标的数量，将处理后的图像加入到训练数据集中。

### 图像分割增加噪声

我们采用SAM（Segment anything model）对原始数据集中的每一张图像进行图像分割来增加图像噪声，将得到的带掩码的图像添加到训练数据集中。

### mosaic

对于原始图像，翻转或旋转后的图像，图像分割后的图像进行mosaic数据增强，对每四张图像，通过尺寸调整与比例缩放，拼接形成一张图像。

## 数据格式转换

为了便于加载数据，运行代码，我们将数据集转换为coco格式。转换格式的运行代码路径为：

```
project/code/data_prepare/2coco.py
```

运行脚本路径为：

```
project/code/data2coco.sh
```

转换后的数据集路径为：

```
project/user_data/cocodataset
```

## 模型训练

我们所使用的基础模型为RT-DETR （[Real-Time Detection Transformer](https://arxiv.org/pdf/2304.08069.pdf)）模型，是一个使用transformer架构的实时检测模型。原模型使用的主干网络为ResNet50,我们使用Focal_Net([Focal Modulation Networks](https://arxiv.org/abs/2203.11926))作为主干网络，并使用在coco数据集上训练完成的模型作为预训练模型来对比赛数据集进行训练。模型训练主要采用paddlepaddle框架。

训练模型基础参数为

训练模型的脚本路径为：

```
project/code/train.sh
```

预训练模型链接为：

https://bj.bcebos.com/v1/paddledet/models/rtdetr_focalnet_L_384_3x_coco.pdparams

下载完毕后放到路径：

project/user_data/model_data/rtdetr_focalnet_L_384_3x_coco.pdparams

## 测试推理

训练完毕后，为了便于提交预测，我们将模型导出为onnx文件，导出的脚本路径为：

```
project/code/export_model.sh
```

其中，导出onnx文件分为两步，第一步将pdparams文件转为中间文件来存储模型的参数，配置及数据预处理信息,这些信息统一保存到一个文件夹中，路径为：

```
project/user_data/model_data/output_inference
```

第二步将中间文件转为onnx文件，onnx文件路径为

```
project/user_data/model_data/rtdetr_focalnet_L_384_3x_coco.onnx
```

使用onnx文件进行最终测试，测试脚本路径为：

```
project/code/test.sh
```

测试完毕后，测试结果按照比赛要求格式保存,路径为：

```
project/prediction_result/output
```

# 文件结构

```
project
├── README.md                # 解决方案及算法介绍文件
├── requirements.txt         # Python环境依赖
├── xfdata                   # 比赛数据集
├── user_data                # 数据文件夹
   ├── aug_dataset              # 数据增强后的数据集
      └── ...
   ├── cocodataset              # 数据增强后的coco格式数据集
      └── ...
   └── model_data               # 模型文件
      ├── rtdetr_focalnet_L_384_3x_coco.pdparams    # 预训练权重，paddle格式
      ├── best_model.pdparams                       # 微调后权重，paddle格式
      ├── rtdetr_focalnet_L_384_3x_coco.onnx        # 微调后权重，onnx格式
      └── output_inference                          # 保存导出onnx时生成的一些中间文件
         └── ...
├── code                     # 代码文件夹
   ├── data_aug.sh           # 数据增强执行文件
   ├── data2coco.sh          # 数据格式转换执行文件
   ├── train.sh              # 模型训练执行文件
   ├── export_model.sh       # 导出onnx模型执行文件
   ├── test.sh               # 预测执行文件
   ├── data_prepare             # 数据预处理代码文件夹
      ├── data_aug.py              # 数据增强代码
      └── 2coco.py                 # 数据格式转换代码
   ├── train                    # 训练代码文件夹
      ├── ...
      └── ...
   ├── test                    # 预测代码文件夹
      └── run.py
├── prediction_result        # 预测结果文件夹
   └── output
         └── ...
```



# 程序运行

注意：本次程序运行均在project/code目录下,否则会有路径问题。

## 脚本正确执行顺序

```
1.数据增强: bash data_aug.sh

2.格式转换: bash data2coco.sh

3.模型训练: bash train.sh

4.导出模型: bash export_model.sh

5.模型测试: bash test.sh
```

## 说明

（1）程序从开始到结束的每一个脚本执行产生的结果文件都已提供，防止程序运行失败或时间过长无法进行下一步，同时在程序运行之前，先把结果文件进行备份，防止被运行程序删除。

（2）在数据增强环节，图像分割运行时间较长（使用显存为6G的RTX3060运行两小时），可选择性执行。

（3）模型训练环节，batch-size设置为4，worker_num设置为4时，对显存的要求较高（使用NVIDIA A40训练显存占用达到40G），调节batch-size路径为

```
project/code/train/rtdetr_paddle/configs/rtdetr/_base_/rtdetr_reader.yml
```

（4）模型训练完毕后，需手动将训练好的模型best_model.pdparams放到model_data中再进行下一步，提供的模型已经放好位置。





