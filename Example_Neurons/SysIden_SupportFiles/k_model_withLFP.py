#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:08:13 2022

@author: jinani

"""
# Import Required Libraries
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from k_layers import gaussian2dMapLayer, PowerLowNonLinearity
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import PReLU
import numpy as np
from tensorflow.keras.regularizers import l2
from k_CustomLayersDN import DivisiveNorm

# Pass 1: Filter estimate pass
def model_pass1(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize, lfp_length, l2_value):
    N_Kern=1
    stride = (Stride,Stride)
    Initial_PReLU=0.5
    Initial_Gaussian_Mean=None
    Initial_Gaussian_Sigma=None
    Initial_Dense_Values=[np.ones((1,1)),np.zeros((1))]

    
    inputLayer =keras.layers.Input(shape=Input_Shape, name="visual_stimuli") 
    
    lfp_data = keras.layers.Input(shape=[lfp_length], name="lfp_data")
    
    model_conv = Conv2D(N_Kern, Filter_Size, 
                                padding='valid',
                                input_shape=Input_Shape,
                                kernel_initializer='glorot_normal',
                                strides = stride)(inputLayer)   
    
    preluWeight = np.array(Initial_PReLU,ndmin=3) 
    model_prelu = PReLU(weights=[preluWeight],shared_axes=[1,2,3])(model_conv)   
    
    model_pool = AveragePooling2D(pool_size=(Pool_Size, Pool_Size))(model_prelu) 
    
    model_gaussian = gaussian2dMapLayer(
            (downsampImageSize,downsampImageSize),
            init_mean=Initial_Gaussian_Mean,
            init_sigma = Initial_Gaussian_Sigma)(model_pool)
    
    lfp_driven_unit = keras.layers.Dense(1,kernel_regularizer=l2(l2_value),name="lfp_driven_unit")(lfp_data)
    
    #additive_Layer = keras.layers.Add()([model_gaussian,lfp_driven_unit])
    #additive_Layer = keras.layers.Multiply()([model_gaussian,lfp_driven_unit])
    additive_Layer = DivisiveNorm()([model_gaussian,lfp_driven_unit])
    
    model_dense5 = Dense((1),weights=Initial_Dense_Values)(additive_Layer) 
    
    output = Activation('relu')(model_dense5) 
    
    model = keras.models.Model(inputs=[inputLayer,lfp_data],outputs =output)
    
    return model


# Pass 2: Power Law Exponent pass
def model_pass2(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize,Initial_Filter_Weights,
                Initial_exp, lfp_length, l2_value):
    
    N_Kern=1
    stride = (Stride,Stride)
    Initial_PReLU=0.5
    Initial_Gaussian_Mean=None
    Initial_Gaussian_Sigma=None
    Initial_Dense_Values=[np.ones((1,1)),np.zeros((1))]
    
    
    inputLayer =keras.layers.Input(shape=Input_Shape, name="visual_stimuli")
    
    lfp_data = keras.layers.Input(shape=[lfp_length], name="lfp_data")
    
    model_conv = Conv2D(N_Kern, Filter_Size, 
                                padding='valid',
                                input_shape=Input_Shape,
                                weights = Initial_Filter_Weights,
                                strides = stride)(inputLayer)   
    
    preluWeight = np.array(Initial_PReLU,ndmin=3) 
    model_prelu = PReLU(weights=[preluWeight],shared_axes=[1,2,3])(model_conv)   
    
    model_pool = AveragePooling2D(pool_size=(Pool_Size, Pool_Size))(model_prelu)
    
    model_gaussian = gaussian2dMapLayer(
            (downsampImageSize,downsampImageSize),
            init_mean=Initial_Gaussian_Mean,
            init_sigma = Initial_Gaussian_Sigma)(model_pool)
    
    lfp_driven_unit = keras.layers.Dense(1,kernel_regularizer=l2(l2_value),name="lfp_driven_unit")(lfp_data)
    
    #additive_Layer = keras.layers.Add()([model_gaussian,lfp_driven_unit])
    #additive_Layer = keras.layers.Multiply()([model_gaussian,lfp_driven_unit])
    additive_Layer = DivisiveNorm()([model_gaussian,lfp_driven_unit])
    
    model_dense5 = Dense((1),weights=Initial_Dense_Values)(additive_Layer) 
    
    output= PowerLowNonLinearity(init_exp=Initial_exp)(model_dense5)
    
    model2 = keras.models.Model(inputs=[inputLayer,lfp_data],outputs =output)
    
    return model2
    
## for sweepwise system iden analysis    
# Pass 2: Power Law Exponent pass
def model_pass2_(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize,
                Initial_exp, lfp_length, l2_value):
    
    N_Kern=1
    stride = (Stride,Stride)
    Initial_PReLU=0.5
    Initial_Gaussian_Mean=None
    Initial_Gaussian_Sigma=None
    Initial_Dense_Values=[np.ones((1,1)),np.zeros((1))]
    
    
    inputLayer =keras.layers.Input(shape=Input_Shape, name="visual_stimuli")
    
    lfp_data = keras.layers.Input(shape=[lfp_length], name="lfp_data")
    
    model_conv = Conv2D(N_Kern, Filter_Size, 
                                padding='valid',
                                input_shape=Input_Shape,
                                kernel_initializer='glorot_normal',
                                strides = stride)(inputLayer)   
    
    preluWeight = np.array(Initial_PReLU,ndmin=3) 
    model_prelu = PReLU(weights=[preluWeight],shared_axes=[1,2,3])(model_conv)   
    
    model_pool = AveragePooling2D(pool_size=(Pool_Size, Pool_Size))(model_prelu)
    
    model_gaussian = gaussian2dMapLayer(
            (downsampImageSize,downsampImageSize),
            init_mean=Initial_Gaussian_Mean,
            init_sigma = Initial_Gaussian_Sigma)(model_pool)
    
    lfp_driven_unit = keras.layers.Dense(1,kernel_regularizer=l2(l2_value),name="lfp_driven_unit")(lfp_data)
    
    #additive_Layer = keras.layers.Add()([model_gaussian,lfp_driven_unit])
    #additive_Layer = keras.layers.Multiply()([model_gaussian,lfp_driven_unit])
    additive_Layer = DivisiveNorm()([model_gaussian,lfp_driven_unit])
    
    model_dense5 = Dense((1),weights=Initial_Dense_Values)(additive_Layer) 
    
    output= PowerLowNonLinearity(init_exp=Initial_exp)(model_dense5)
    
    model2 = keras.models.Model(inputs=[inputLayer,lfp_data],outputs =output)
    
    return model2
        
    
    
    