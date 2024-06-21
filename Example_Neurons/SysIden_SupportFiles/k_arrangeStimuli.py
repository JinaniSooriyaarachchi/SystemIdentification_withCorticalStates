#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 08:47:38 2022

@author: jinani
"""
import numpy as np
import sys

sys.path.insert(0,'/home/jinani/Desktop/Paper1_github/SysIden_SupportFiles/') # path to the support files folder
from k_functions import buildCalcAndScale, kConvNetStyle

# arrange_stimuli: to arrange stimuli dataset to be used in system identification
def arrange_stimuli(Stimuli_dataset,num_timelags):
    
    Training_stimuli=Stimuli_dataset['Training_stimuli']  # 480 x 480 x 7500
    Validation_stimuli=Stimuli_dataset['Validation_stimuli']  # 480 x 480 x 1875
    Testing_stimuli=Stimuli_dataset['Testing_stimuli']  # 480 x 480 x 1875
    
    # take 5 sweeps of data
    Training_stimuli_allTrials=None
    for i in range(5):
        if i==0:
            Training_stimuli_allTrials=Training_stimuli
        else:
            Training_stimuli_allTrials=np.dstack((Training_stimuli_allTrials,Training_stimuli))
            
    Validation_stimuli_allTrials=None
    for i in range(5):
        if i==0:
            Validation_stimuli_allTrials=Validation_stimuli
        else:
            Validation_stimuli_allTrials=np.dstack((Validation_stimuli_allTrials,Validation_stimuli))
            
    Testing_stimuli_allTrials=None
    for i in range(5):
        if i==0:
            Testing_stimuli_allTrials=Testing_stimuli
        else:
            Testing_stimuli_allTrials=np.dstack((Testing_stimuli_allTrials,Testing_stimuli))
    
    # Get movie and image sizes
    movieSize = np.shape(Training_stimuli_allTrials)       # 30 x 30 x 7500
    imSize = (movieSize[0],movieSize[1])   # 30 x 30
    
    # Reshape stimuli
    Training_stimuli_allTrials = np.reshape(Training_stimuli_allTrials,(imSize[0]*imSize[1],movieSize[2]))  # 900 x 7500
    Training_stimuli_allTrials = np.transpose(Training_stimuli_allTrials)  # 7500 x 900
    
    Validation_stimuli_allTrials = np.reshape(Validation_stimuli_allTrials,(imSize[0]*imSize[1],np.shape(Validation_stimuli_allTrials)[2]))  # 900 x 1875
    Validation_stimuli_allTrials = np.transpose(Validation_stimuli_allTrials)  # 1875 x 900
    
    Testing_stimuli_allTrials = np.reshape(Testing_stimuli_allTrials,(imSize[0]*imSize[1],np.shape(Testing_stimuli_allTrials)[2]))  # 900 x 1875
    Testing_stimuli_allTrials = np.transpose(Testing_stimuli_allTrials)  # 1875 x 900
    
    # Normalize data
    Training_stimuli_allTrials = buildCalcAndScale(Training_stimuli_allTrials)   # 7500 x 900
    Validation_stimuli_allTrials = buildCalcAndScale(Validation_stimuli_allTrials)   # 1875 x 900
    Testing_stimuli_allTrials = buildCalcAndScale(Testing_stimuli_allTrials) #1875 x 900
    
    # Arrange time lagged dataset
    Frames =list(range(num_timelags))
    estSet = kConvNetStyle(Training_stimuli_allTrials,Frames)    # 7500 x 30 X 30 x lags
    regSet = kConvNetStyle(Validation_stimuli_allTrials,Frames)    # 1875 x 30 X 30 x lags
    predSet = kConvNetStyle(Testing_stimuli_allTrials,Frames)   # 1875 x 30 X 30 x lags
            
    return estSet,regSet,predSet,imSize