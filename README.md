# Keras-UNet
基于Keras搭建一个简单的unet，用医学图像数据集对unet进行训练，完成模型的保存和加载和图像分割测试。

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />
注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：保存数据集文件<br />
2. /save_models：保存训练好的模型权重文件<br /><br />

UNet：<br />
UNet 属于 FCN 的一种变体，它可以说是最常用、最简单的一种分割模型。<br />
UNet 在2015 年被提出，它简单、高效、易懂、容易构建，且可以从小数据集中训练。<br />
UNet 的初衷是为了解决医学图像分割的问题，在解决细胞层面的分割的任务方面，其在 2015 年的 ISBI cell tracking 比赛中获得了多个第一。<br />
之后，UNet 凭借其突出的分割效果而被广泛应用在语义分割的各个方向（如卫星图像分割，工业瑕疵检测等）。<br /><br />
其网络结构是对称的，形似英文字母 U，故而被称为 UNet 。就整体而言，<br />
UNet 是一个Encoder-Decoder的结构（与 FCN 相同），前半部分是特征提取，后半部分是上采样。<br /><br />

数据集：<br />
1. Medical_Datasets：医学影像数据集,训练集/测试集包含25/5/原始图像和分割图像<br />
链接：https://pan.baidu.com/s/1El3IXK4-ycQVCaXBYurVsA?pwd=52dl 提取码：52dl<br />

使用Keras预处理工具 ImageDataGenerator，对数据集中原始图片进行缩放，旋转等操作，以增强数据<br />
使用flow_from_directory()方法从数据集的子目录中实时生成训练和测试数据<br /><br />

损失函数：<br />
Dice 损失函数<br />
评价指标：<br />
Dice 系数<br />
