#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:08:13 2022

@author: jinani

"""
# Import Required Libraries
import sys
sys.path.insert(0,'/home/jinani/Desktop/SYSIDEN_CODE/SysIden_SupportFiles/') # path to the support files folder
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from k_layers import gaussian2dMapLayer, PowerLowNonLinearity
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import PReLU
import numpy as np


# Pass 1: Filter estimate pass
def model_pass1(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize):
    N_Kern=1
    stride = (Stride,Stride)
    Initial_PReLU=0.5
    Initial_Gaussian_Mean=None
    Initial_Gaussian_Sigma=None
    Initial_Dense_Values=[np.ones((1,1)),np.zeros((1))]

    inputLayer =keras.layers.Input(shape=Input_Shape, name="visual_stimuli")    
    
    model_conv = Conv2D(N_Kern, Filter_Size, 
                                padding='valid',
                                input_shape=Input_Shape,
                                kernel_initializer='glorot_normal',
                                strides = stride)(inputLayer)   
    
    preluWeight = np.array(Initial_PReLU,ndmin=3) 
    model_prelu = PReLU(weights=[preluWeight],shared_axes=[1,2,3])(model_conv)   # s x 18 x 18 x 1
    
    model_pool = AveragePooling2D(pool_size=(Pool_Size, Pool_Size))(model_prelu) # s x 9 x 9 x 1
    
    model_gaussian = gaussian2dMapLayer(
            (downsampImageSize,downsampImageSize),
            init_mean=Initial_Gaussian_Mean,
            init_sigma = Initial_Gaussian_Sigma)(model_pool)   # s x 1
    
    model_dense = Dense((1),weights=Initial_Dense_Values)(model_gaussian)  # s x 1
    
    output = Activation('relu')(model_dense)   # s x 1
    
    model = keras.models.Model(inputs=inputLayer,outputs =output)
    
    return model


# Pass 2: Power Law Exponent pass
def model_pass2(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize,Initial_Filter_Weights,Initial_exp):
    
    N_Kern=1
    stride = (Stride,Stride)
    Initial_PReLU=0.5
    Initial_Gaussian_Mean=None
    Initial_Gaussian_Sigma=None
    Initial_Dense_Values=[np.ones((1,1)),np.zeros((1))]
    
    inputLayer =keras.layers.Input(shape=Input_Shape, name="visual_stimuli")    
    
    model_conv = Conv2D(N_Kern, Filter_Size, 
                                padding='valid',
                                input_shape=Input_Shape,
                                weights = Initial_Filter_Weights,
                                strides = stride)(inputLayer)   
    
    preluWeight = np.array(Initial_PReLU,ndmin=3) 
    model_prelu = PReLU(weights=[preluWeight],shared_axes=[1,2,3])(model_conv)   # s x 18 x 18 x 1
    
    model_pool = AveragePooling2D(pool_size=(Pool_Size, Pool_Size))(model_prelu) # s x 9 x 9 x 1
    
    model_gaussian = gaussian2dMapLayer(
            (downsampImageSize,downsampImageSize),
            init_mean=Initial_Gaussian_Mean,
            init_sigma = Initial_Gaussian_Sigma)(model_pool)   # s x 1
    
    model_dense = Dense((1),weights=Initial_Dense_Values)(model_gaussian)  # s x 1
    
    output= PowerLowNonLinearity(init_exp=Initial_exp)(model_dense)
    
    model2 = keras.models.Model(inputs=inputLayer,outputs =output)
    
    return model2
    
    
    
    
    
    