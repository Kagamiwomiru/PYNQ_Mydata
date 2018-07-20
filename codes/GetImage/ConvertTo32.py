# coding: utf-8
# # convertTo32
# 画像を32x32に一括変換します。
from PIL import Image
import glob
import os

#dir->faces/
def convertTo32(dir):

    for sub_dir in glob.glob(dir+str('*')): #/home/kagamiwomiru/MakeCifar/dir/Kagami,Uchiyama/
        for data in glob.glob(sub_dir+str('/*.jpg')): #Kagami/nuga00.jpg,....
            img=Image.open(data)
            img_resize=img.resize((32,32))
            img_resize.save(data)
