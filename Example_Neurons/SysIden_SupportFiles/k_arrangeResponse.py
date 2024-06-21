#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 08:51:17 2022

@author: jinani
"""
import scipy.io as sio
import numpy as np

# 5. arrange_responses: to arrange response dataset
def arrange_responses(dataset_path,dataset_name):
    
    Train_response = sio.loadmat(dataset_path+dataset_name + '_estSetResp.mat')['est_resp'] 
    y_est=None
    for i in range(5):
        if i==0:
            y_est=Train_response[:,i]
        else:
            y_est=np.hstack((y_est,Train_response[:,i]))
              
    Valid_response=sio.loadmat(dataset_path+dataset_name + '_regSetResp.mat')['reg_resp'] 
    y_reg=None
    for i in range(5):
        if i==0:
            y_reg=Valid_response[:,i*4]
        else:
            y_reg=np.hstack((y_reg,Valid_response[:,i*4]))
        
    Test_response=sio.loadmat(dataset_path+dataset_name + '_predSetResp.mat')['pred_resp'] 
    y_pred=None
    for i in range(5):
        if i==0:
            y_pred=Test_response[:,i*4]
        else:
            y_pred=np.hstack((y_pred,Test_response[:,i*4]))
        
    return y_est,y_reg,y_pred