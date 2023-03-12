# coding=gbk


from keras.layers import Input, Conv2D, MaxPooling2D, Layer
from keras.layers import Concatenate, UpSampling2D, BatchNormalization
from keras.models import Model


'''
UNet 属于 FCN 的一种变体，它可以说是最常用、最简单的一种分割模型，
它简单、高效、易懂、容易构建，且可以从小数据集中训练。2015 年被提出 。
UNet 的初衷是为了解决医学图像分割的问题，在解决细胞层面的分割的任务方面，
其在 2015 年的 ISBI cell tracking 比赛中获得了多个第一。
之后，UNet 凭借其突出的分割效果而被广泛应用在语义分割的各个方向（如卫星图像分割，工业瑕疵检测等）

其网络结构是对称的，形似英文字母 U，故而被称为 UNet 。就整体而言，
UNet 是一个Encoder-Decoder的结构（与 FCN 相同），前半部分是特征提取，后半部分是上采样。

'''

class DownBlock(Layer):
    '''下采样模块'''
    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(width, 3, activation='relu', padding='same')
        self.conv2 = Conv2D(width, 3, activation='relu', padding='same')
        self.maxp = MaxPooling2D()
        self.bn = BatchNormalization()

    def call(self, inputs):
        x = inputs
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.maxp(x), x


class UpBlock(Layer):
    '''下采样模块'''
    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.up_sample = UpSampling2D(interpolation="nearest")
        self.conv1 = Conv2D(width, 3, activation='relu', padding='same')
        self.conv2 = Conv2D(width, 3, activation='relu', padding='same')
        self.conc = Concatenate()
        self.bn = BatchNormalization()

    def call(self, inputs):
        x, skip = inputs
        x = self.up_sample(x)
        x = self.conc([x, skip])
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class UNet(Model):
    '''unet的实现
        skips：保存下采样模块的输出，连接到上采样模块 
        down_blocks：存放多个下采样模块
        conv_block：上采样和下采样过程中过度处卷积模块
        up_blocks：存放多个上采样模块
        out：处理最后上采样的结果，匹配标签维度
    '''
    def __init__(self, output_channel, **kwargs):
        super().__init__(**kwargs)
        self.skips = []     
        self.down_blocks = [
            DownBlock(32),
            DownBlock(64),
            DownBlock(128)]
        self.conv_block = [
            Conv2D(256, 3, activation='relu', padding='same'),
            Conv2D(128, 3, activation='relu', padding='same')
            ]
        self.up_blocks = [
            UpBlock(128),
            UpBlock(64),
            UpBlock(32)
            ]
        self.out = Conv2D(output_channel, 1, activation='sigmoid')

    def call(self, inputs):
        x = inputs
        for block in self.down_blocks:
            x, skip = block(x)
            self.skips.append(skip)

        for block in self.conv_block:
            x = block(x)

        for block in self.up_blocks:
            skip = self.skips.pop()
            x = block([x, skip])
        return self.out(x)

    def build(self, input_shape):
        super().build(input_shape)
        self.call(Input(input_shape[1:]))






