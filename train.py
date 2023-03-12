#coding=gbk

from model import UNet
from loss import dice_coef, dice_coef_loss
from utils import *


#------------------------------���ò���----------------------------------------
img_heigh = 256
img_width = 256
img_channel = 1
batch_size = 8
epochs = 20
steps_per_epoch=500
#-----------------------------------------------------------------------------


#--------------------------------·��------------------------------------------
data_path = "D:/����/python����/�����ֲ�/UNet/datasets/Medical_Datasets/train/"
save_path = "D:/����/python����/�����ֲ�/UNet/save_models/unet.h5"
#-----------------------------------------------------------------------------


#------------------------------��������----------------------------------------
myGene = trainGenerator(
    batch_size, 
    data_path, 
    image_folder='Images', 
    mask_folder='Labels', 
    target_size=(img_heigh, img_width))
#-----------------------------------------------------------------------------


#------------------------------����ģ��----------------------------------------
unet = UNet(img_channel)
unet.build((None, img_heigh, img_width, img_channel))
unet.compile(
    optimizer='adam',
    loss=dice_coef_loss,
    metrics=[dice_coef]
    )
unet.summary()
#----------------------------------------------------------------------------


#-----------------------------ѵ������ģ��-------------------------------------
unet.fit(myGene, steps_per_epoch=steps_per_epoch, epochs=epochs)
unet.save_weights(save_path)
#----------------------------------------------------------------------------