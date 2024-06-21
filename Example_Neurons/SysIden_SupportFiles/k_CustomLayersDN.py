#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:55:20 2021

@author: jinani

This is the code for the custom Divisive normalization Layer.
The layer have two trainable parameters:
    Semi-saturation constant
    Gain of brainstate

"""

from tensorflow import keras
from tensorflow.keras import initializers
import numpy as np
import tensorflow.keras.backend as K

class DivisiveNorm(keras.layers.Layer):
    def __init__(self, init = 'zero', **kwargs):
        super().__init__(**kwargs)
        self.init = initializers.get(init) 
        self.tolerance = np.asarray(0.01) #use tolerance from stopping division by zero
        semiSat=np.asarray([2.5])  # initial guesses
        gain=np.asarray([0.25])  # initial guesses
        self.init_semiSat = semiSat.astype('float32')
        self.init_gain = gain.astype('float32')

    def build(self, input_shape):
        #print(input_shape)
        self.semiSat = self.add_weight(shape=(1,),  
                                    initializer=self.init,
                                    name='semiSat')
        self.gain = self.add_weight(shape=(1,),
                            initializer=self.init,
                            name='gain')
        if self.init_semiSat is not None:
            K.set_value(self.semiSat,self.init_semiSat)
            del self.init_semiSat
        if self.init_gain is not None:
            K.set_value(self.gain,self.init_gain)
            del self.init_gain
        
        self.built = True
    
    def call(self, X):
        return X[0]/(self.semiSat+self.gain*X[1]+self.tolerance)  # DN equation
    
    def compute_output_shape(self):
        return (1,1)
    
    def get_config(self):
        base_config = super(DivisiveNorm, self).get_config()
        return dict(list(base_config.items()))
        
        
        
        
