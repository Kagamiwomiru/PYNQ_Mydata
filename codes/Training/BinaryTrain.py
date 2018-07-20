#BSD 3-Clause License
# -*- coding: utf-8 -*-
#=======
#
#Copyright (c) 2017, Xilinx Inc.
#All rights reserved.
#
#Based Matthieu Courbariaux's CIFAR-10 example code
#Copyright (c) 2015-2016, Matthieu Courbariaux
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the copyright holder nor the names of its 
#      contributors may be used to endorse or promote products derived from 
#      this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
#EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# CUCUMBER9を元にデータセットではなく、データのディレクトリから学習できるようにします

from __future__ import print_function
import sys
import os
import time
import numpy as np
np.random.seed(1234) # for reproducibility?
import theano
import theano.tensor as T
import lasagne
import cPickle as pickle
import gzip
# パスの追加が必要
sys.path.append('/home/kagamiwomiru/Training/')
import binary_net
import cnv
from collections import OrderedDict
from pylearn2.datasets import DenseDesignMatrix
import glob
from PIL import Image

# データセット設定
#学習用、バリデーション用画像パス
Train_dir='/home/kagamiwomiru/MakeCifar/train_face/'
Test_dir='/home/kagamiwomiru/MakeCifar/eval_face'

#学習画像枚数
nb_train_samples=900
#テスト画像枚数
nb_test_samples=30
#ラベル
labels=['Kagami','Uchiyama','Kato']

# BinaryNetパラメータ設定
#バッチサイズ
batch_size = 5
# alphaは指数移動平均係数
LP_alpha = .1
LP_epsilon = 1e-4

# "Glorot" means we are using the coefficients from Glorot's paper
LP_W_LR_scale = "Glorot" 

# 学習パラメータ設定
#エポック数
num_epochs = 100
# Decaying 学習率
LR_start = 0.001
LR_fin = 0.0000003
#画像のシャッフル
shuffle_parts = 1       

#学習結果（学習済みモデル）の保存先    
save_path = "parameters.npz"
    


class DATASET(DenseDesignMatrix):
    #クラスの初期化
    def __init__(self,which_set):
        # self.train_path='/home/kagamiwomiru/MakeCifar/train_face/'
        self.train_path=Train_dir
        # self.test_path='/home/kagamiwomiru/MakeCifar/eval_face'
        self.test_path=Test_dir
        self.img_shape=(3,32,32)
        #配列要素の積を出す
        self.img_size=np.prod(self.img_shape)
        #学習かバリデーションかによって画像の読み込み先を変えてる
        if which_set in {'train'}:
            X,y=self.load_data()
        elif which_set in {'test','valid'}:
            X,y=self.load_test_data()
        
        X=X.astype('float32')

        super(DATASET,self).__init__(X=X,y=y)
    
            
    #trainデータを読み込みます。変更前はデータセットからでしたが今回は生の画像データから読み込むようにします。
    def load_data(self):
        #学習画像枚数
        # nb_train_samples=900
        i=0
        # labels=['Kagami','Uchiyama','Kato']
        X=np.zeros((nb_train_samples,self.img_size),dtype='uint8')#画像データ？※1
        y=np.zeros((nb_train_samples,1),dtype='uint8') #ラベル？
        #////画像データとラベルの読み込み////#
        for label in labels:
            fpath=os.path.join(self.train_path,label)#fpath=~/MakeCifar10/train_face/Kagami/以下
            filepath=fpath+str('/*.jpg')
            for image in glob.glob(filepath):
                im=np.array(Image.open(image))
                X[i,:]=im.reshape(3072)
                if(label=='Kagami'):
                    y[i,0]=0
                elif(label=='Uchiyama'):
                    y[i,0]=1
                elif(label=='Kato'):
                    y[i,0]=2
                i+=1
        return X,y
        
    #testデータを読み込みます。変更前はデータセットからでしたが今回は生の画像データから読み込むようにします。
    def load_test_data(self):
        #テスト画像枚数
        # nb_test_samples=30
        i=0
        # labels=['Kagami','Uchiyama','Kato']
        X=np.zeros((nb_test_samples,self.img_size),dtype='uint8')#画像データ？※1
        y=np.zeros((nb_test_samples,1),dtype='uint8') #ラベル？
        #////画像データとラベルの読み込み////#
        for label in labels:
            fpath=os.path.join(self.test_path,label)#fpath=~/MakeCifar10/train_face/Kagami/以下
            filepath=fpath+str('/*.jpg')
            for image in glob.glob(filepath):
                im=np.array(Image.open(image))
                X[i,:]=im.reshape(3072)
                if(label=='Kagami'):
                    y[i,0]=0
                elif(label=='Uchiyama'):
                    y[i,0]=1
                elif(label=='Kato'):
                    y[i,0]=2
                i+=1
        return X,y

if __name__ =="__main__":
    learning_parameters=OrderedDict()

    # BinaryNetパラメータ
    # batch_size = 5
    print("batch_size = "+str(batch_size))

    # alpha is the exponential moving average factor
    # alphaは指数移動平均係数
    learning_parameters.alpha = LP_alpha
    print("alpha = "+str(learning_parameters.alpha))
    learning_parameters.epsilon = LP_epsilon
    print("epsilon = "+str(learning_parameters.epsilon))
        
    # # W_LR_scale = 1.    
    learning_parameters.W_LR_scale = LP_W_LR_scale
    print("W_LR_scale = "+str(learning_parameters.W_LR_scale))


    # Training parameters
    # num_epochs = 100
    print("エポック数 = "+str(num_epochs))


    # Decaying LR 
    # LR_start = 0.001
    print("学習率_開始時 = "+str(LR_start))
    # LR_fin = 0.0000003
    print("学習率_終了時 = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("学習率_減衰 = "+str(LR_decay))

    #パラメータの保存先    
    # save_path = "parameters.npz"
    print("保存先 = "+str(save_path))



    #nb_train_samplesの値と揃える。
    # train_set_size = 900
    train_set_size=nb_train_samples
    print("train_set_size = "+str(train_set_size))
    # shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    print('データセット読み込み開始...')

    train_set=DATASET(which_set="train")
    valid_set=DATASET(which_set="valid")
    test_set=DATASET(which_set="test")

    # bc01 format
    train_set.X = np.reshape(np.subtract(np.multiply(2./255.,train_set.X),1.),(-1,3,32,32))
    valid_set.X = np.reshape(np.subtract(np.multiply(2./255.,valid_set.X),1.),(-1,3,32,32))
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))

    #ターゲットをflatten
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)

    # Onehot the targets
    # ターゲットをonehot表現に変換
    train_set.y = np.float32(np.eye(3)[train_set.y])    
    valid_set.y = np.float32(np.eye(3)[valid_set.y])
    test_set.y = np.float32(np.eye(3)[test_set.y])

    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    print('CNNを構築中...') 

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = cnv.genCnv(input, 3, learning_parameters)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)

    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))


    # W updates
    W = lasagne.layers.get_all_params(cnn, binary=True)
    W_grads = binary_net.compute_grads(loss,cnn)
    updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
    updates = binary_net.clipping_scaling(updates,cnn)

    # other parameters updates
    params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
    updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)


    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('学習中...')

    binary_net.train(
            train_fn,val_fn,
            cnn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y,
            save_path=save_path,
            shuffle_parts=shuffle_parts)