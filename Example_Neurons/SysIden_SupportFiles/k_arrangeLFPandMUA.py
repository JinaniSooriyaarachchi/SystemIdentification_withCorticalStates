#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:04:47 2022

@author: jinani
"""
import h5py
import numpy as np
from k_functions import buildCalcAndScale

def arrange_LFP(LFPpath,Channel, start_ch, end_ch):
    
    lfp_path=LFPpath
    #LFP_Train=h5py.File('allCh_LFPTrainingDataset_combined_notch_model2.mat')['allCh_TrainingDataset_combined']
    LFP_Train=h5py.File(lfp_path+'allCh_LFPTrainingDataset_combined.mat')['allCh_TrainingDataset_combined']
    LFP_Train1=np.transpose(LFP_Train)
    LFP_Train=np.delete(LFP_Train1,np.s_[(Channel-1)*150:Channel*150],axis=2)
    Trial1_Train_LFP=None
    for i in range(5):
        if i==0:
            Trial1_Train_LFP=LFP_Train[i,:,(start_ch-1)*150:(end_ch-1)*150]  
        else:
            Trial1_Train_LFP=np.vstack((Trial1_Train_LFP,LFP_Train[i,:,(start_ch-1)*150:(end_ch-1)*150]))
            
    
    #LFP_Valid=h5py.File('allCh_LFPValidationDataset_combined_notch_model2.mat')['allCh_ValidationDataset_combined']
    LFP_Valid=h5py.File(lfp_path+'allCh_LFPValidationDataset_combined.mat')['allCh_ValidationDataset_combined']
    LFP_Valid1=np.transpose(LFP_Valid)
    LFP_Valid=np.delete(LFP_Valid1,np.s_[(Channel-1)*150:Channel*150],axis=2)
    Trial1_Valid_LFP=None
    for i in range(5):
        if i==0:
            Trial1_Valid_LFP=LFP_Valid[i*4,:,(start_ch-1)*150:(end_ch-1)*150]
        else:
            Trial1_Valid_LFP=np.vstack((Trial1_Valid_LFP,LFP_Valid[i*4,:,(start_ch-1)*150:(end_ch-1)*150]))
            
            
    #LFP_Test=h5py.File('allCh_LFPTestingDataset_combined_notch_model2.mat')['allCh_TestingDataset_combined']
    LFP_Test=h5py.File(lfp_path+'allCh_LFPTestingDataset_combined.mat')['allCh_TestingDataset_combined']
    LFP_Test1=np.transpose(LFP_Test)
    LFP_Test=np.delete(LFP_Test1,np.s_[(Channel-1)*150:Channel*150],axis=2)
    Trial1_Test_LFP=None
    for i in range(5):
        if i==0:
            Trial1_Test_LFP=LFP_Test[i*4,:,(start_ch-1)*150:(end_ch-1)*150]
        else:
            Trial1_Test_LFP=np.vstack((Trial1_Test_LFP,LFP_Test[i*4,:,(start_ch-1)*150:(end_ch-1)*150]))
    
    estLFP=buildCalcAndScale(Trial1_Train_LFP)  
    regLFP=buildCalcAndScale(Trial1_Valid_LFP)  
    predLFP=buildCalcAndScale(Trial1_Test_LFP) 

    return  estLFP, regLFP, predLFP  


def arrange_MUA (MUApath,Channel):
    
    MUA_Train=h5py.File(MUApath+'allCh_MUATrainingDataset_combined.mat')['allCh_TrainingDataset_combined']
    MUA_Train1=np.transpose(MUA_Train)
    MUA_Train=np.delete(MUA_Train1,np.s_[(Channel-1)*150:Channel*150],axis=2)
    Trial1_Train_MUA=None
    for i in range(5):
        if i==0:
            Trial1_Train_MUA=MUA_Train[i,:,(Channel-2)*150:Channel*150] 
        else:
            Trial1_Train_MUA=np.vstack((Trial1_Train_MUA,MUA_Train[i,:,(Channel-2)*150:Channel*150]))
            
    
    MUA_Valid=h5py.File(MUApath+'allCh_MUAValidationDataset_combined.mat')['allCh_ValidationDataset_combined']
    MUA_Valid1=np.transpose(MUA_Valid)
    MUA_Valid=np.delete(MUA_Valid1,np.s_[(Channel-1)*150:Channel*150],axis=2)
    Trial1_Valid_MUA=None
    for i in range(5):
        if i==0:
            Trial1_Valid_MUA=MUA_Valid[i*4,:,(Channel-2)*150:Channel*150]
        else:
            Trial1_Valid_MUA=np.vstack((Trial1_Valid_MUA,MUA_Valid[i*4,:,(Channel-2)*150:Channel*150]))
            
            
    MUA_Test=h5py.File(MUApath+'allCh_MUATestingDataset_combined.mat')['allCh_TestingDataset_combined']
    MUA_Test1=np.transpose(MUA_Test)
    MUA_Test=np.delete(MUA_Test1,np.s_[(Channel-1)*150:Channel*150],axis=2)
    Trial1_Test_MUA=None
    for i in range(5):
        if i==0:
            Trial1_Test_MUA=MUA_Test[i*4,:,(Channel-2)*150:Channel*150]
        else:
            Trial1_Test_MUA=np.vstack((Trial1_Test_MUA,MUA_Test[i*4,:,(Channel-2)*150:Channel*150]))

    estMUA=buildCalcAndScale(Trial1_Train_MUA)  
    regMUA=buildCalcAndScale(Trial1_Valid_MUA)  
    predMUA=buildCalcAndScale(Trial1_Test_MUA) 
    
    return  estMUA, regMUA, predMUA  
    
    
    
