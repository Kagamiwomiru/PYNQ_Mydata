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

from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_net
import cnv

from pylearn2.datasets.zca_dataset import ZCA_Dataset   
from pylearn2.datasets.cifar10 import CIFAR10 
from pylearn2.utils import serial

from collections import OrderedDict

from pylearn2.datasets import DenseDesignMatrix


class CUCUMBER9(DenseDesignMatrix):
    def __init__(self,which_set):

        self.path='/home/kagamiwomiru/CUCUMBER-9/prototype_1'

        self.img_shape=(3,32,32)
        self.img_size=np.prod(self.img_shape)

        #学習かバリデーションかによって画像の読み込み先を変えてる
        if which_set in {'train'}:
            X,y=self.load_data()
        elif which_set in {'test','valid'}:
            X,y=self.load_data_test()
        
        X=X.astype('float32')

        super(CUCUMBER9,self).__init__(X=X,y=y)

#Pythonバージョンによって文字コードの指定変えてる
    def unpickle(self,file_name):
        with open(file_name,'rb') as f:
            if sys.version_info.major==2:
                return pickle.load(f)
            elif sys.vesion_info.major==3:
                return pickle.load(f,encoding='latin-1')
    
    def load_data(self):
        #学習枚数
        nb_train_samples=2475

        X=np.zeros((nb_train_samples,self.img_size),dtype='uint8')
        y=np.zeros((nb_train_samples,1),dtype='uint8')

        for i in range(1,6):
            fpath=os.path.join(self.path,'data_batch_'+str(i))
            batch_dict=self.unpickle(fpath)
            data=batch_dict['data']
            labels=batch_dict['labels']

            X[(i-1)*495:i*495, :]=data.reshape(495,self.img_size)
            y[(i-1)*495:i*495, 0]=labels

        return X,y


    def load_data_test(self):
        #テスト用画像枚数
        nb_test_samples=495

        fpath=os.path.join(self.path,'test_batch')
        batch_dict=self.unpickle(fpath)
        data=batch_dict['data']
        labels=batch_dict['labels']

        X=np.zeros((nb_test_samples,self.img_size),dtype='uint8')
        y=np.zeros((nb_test_samples,1),dtype='uint8')
        
        X=data.reshape(nb_test_samples,self.img_size)
        y[:,0]=labels

        return X,y




if __name__ == "__main__":
    
    learning_parameters = OrderedDict()
    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    learning_parameters.alpha = .1
    print("alpha = "+str(learning_parameters.alpha))
    learning_parameters.epsilon = 1e-4
    print("epsilon = "+str(learning_parameters.epsilon))
    
    # W_LR_scale = 1.    
    learning_parameters.W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(learning_parameters.W_LR_scale))
    
    # Training parameters
    num_epochs = 500
    print("エポック数 = "+str(num_epochs))
    
    # Decaying LR 
    LR_start = 0.001
    print("学習率_開始時 = "+str(LR_start))
    LR_fin = 0.0000003
    print("学習率_終了時 = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("学習率_減衰 = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    #パラメータの保存先    
    save_path = "cucumber9_parameters.npz"
    print("保存先 = "+str(save_path))

    #nb_train_samplesの値と揃える。
    train_set_size = 2475
    print("train_set_size = "+str(train_set_size))
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('CUCUMBER9データセットを読み込み中...')
    
    train_set = CUCUMBER9(which_set="train")
    valid_set = CUCUMBER9(which_set="valid")
    test_set = CUCUMBER9(which_set="test")
        
    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    train_set.X = np.reshape(np.subtract(np.multiply(2./255.,train_set.X),1.),(-1,3,32,32))
    valid_set.X = np.reshape(np.subtract(np.multiply(2./255.,valid_set.X),1.),(-1,3,32,32))
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
    
    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    # Onehot the targets
    train_set.y = np.float32(np.eye(9)[train_set.y])    
    valid_set.y = np.float32(np.eye(9)[valid_set.y])
    test_set.y = np.float32(np.eye(9)[test_set.y])
    
    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    print('CNNを構築中...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = cnv.genCnv(input, 9, learning_parameters)

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
