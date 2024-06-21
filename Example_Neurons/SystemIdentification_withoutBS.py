#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:42:31 2022

@author: jinani

This code runs system identification without incorporating the brain state pathway.
Needs to run (from SysIden_SupportFiles folder):
    k_arrangeStimuli.py, 
    k_arrangeResponse.py, 
    k_functions.py,
    k_model_withoutBS.py

"""

# Import Required Libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
from IPython import get_ipython
get_ipython().magic('reset -sf') # To Clear Variables Before Script Runs

import os
import shutil
import scipy.io as sio
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy
import sys

# Import Custom Written Functions and Model
sys.path.insert(0,'/home/jinani/Desktop/Paper1_github/Example_Neurons/SysIden_SupportFiles/') # path to the support files folder
from k_arrangeStimuli import arrange_stimuli
from k_arrangeResponse import arrange_responses
from k_functions import conv_output_length,plotGaussMap,plotReconstruction
from k_model_withoutBS import model_pass1, model_pass2

####################################################
############### CHANGE HERE ########################
# INPUT DETAILS HERE !

dataset_name='H6903.013_1_Ch16' #'H6903.013_1_Ch16' for neuron1 and 'H6903.013_1_Ch28' for neuron2
dataset_path='/home/jinani/Desktop/Paper1_github/Example_Neurons/'+dataset_name+'/'


#neuron1
Stimuli_dataset = sio.loadmat('/home/jinani/Desktop/Paper1_github/Example_Neurons/StimuliData/StimuliData_H6903.013_1_Ch16.mat') # neuron1 
##neuron2
#Stimuli_dataset = sio.loadmat('/home/jinani/Desktop/Paper1_github/Example_Neurons/StimuliData/StimuliData_H6903.013_1_Ch28.mat') # neuron2 

Results_folder='Results_WithoutBS/'         


Filter_Size=15
Pool_Size=1 
imSize=(30,30)


####################################################

dir = dataset_path+Results_folder
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

sys.path.insert(0,'/home/jinani/Desktop/Paper1_github/Example_Neurons/'+dataset_name) 

num_timelags=7
Stride=1

# Arrange stimuli into time lags
estSet,regSet,predSet,imSize=arrange_stimuli(Stimuli_dataset,num_timelags)

# Arrange the Response Dataset
y_est,y_reg,y_pred=arrange_responses(dataset_path,dataset_name)


# Calculate Shape of the Main Input and Intermediate Layer Inputs 
Input_Shape=estSet.shape[1:]   
numRows =  Input_Shape[0]  
numCols =  Input_Shape[1]  
assert numRows == numCols
numFrames = Input_Shape[2] 

convImageSize = conv_output_length(numRows,Filter_Size,'valid',Stride) # Input to Conv2D Layer
downsampImageSize = conv_output_length(convImageSize,Pool_Size,'valid',Pool_Size) # Input to Gaussian Map Layer

# Model for Pass 1 : Filter Estimate Pass
model=model_pass1(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize)
model.summary()
optimizerFunction = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizerFunction)    
earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
history=model.fit(estSet,y_est, validation_data=(regSet,y_reg), epochs = 500,
              batch_size=750,callbacks=[earlyStop,mc],verbose=1)

# Pass 1 Trained Weights
weights = model.get_weights()

# Calculate VAF
predicted_test_response = model.predict(predSet)
predicted_test_response1 = predicted_test_response.reshape(-1)
respTest=y_pred.reshape(-1)
R=np.corrcoef(predicted_test_response1,respTest)
diag=R[0,1]
VAF_test1=diag*diag*100
print (VAF_test1)

# Plot Results
# 1. Learning Curve
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Learning Curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid()
plt.savefig(dataset_path+Results_folder+"1.png")

# 2. Actual/Predicted Response Data
plt.figure(2)
plt.plot(respTest[0:100],color='r',label='Actual')
plt.plot(predicted_test_response[0:100],color='b',label='Estimated')
plt.legend(loc='upper right')
plt.grid()
plt.title("Response Data")
plt.savefig(dataset_path+Results_folder+"2.png")

# 3. PReLU
plt.figure(3)
alpha1 = np.squeeze(weights[2])
x = np.arange(-100,101)
y = np.arange(-100,101)
y[y<=0] = alpha1*y[y<=0] 
plt.plot(x,y)
plt.title('PReLU, alpha = {}'.format(np.round(alpha1,2)))
plt.savefig(dataset_path+Results_folder+"3.png")

# 4. Gaussian Map
plt.figure(4)
mapMean = weights[3]
mapSigma = weights[4]
mapVals = plotGaussMap(mapMean,mapSigma,downsampImageSize)
plt.title('Gaussian Map')
plt.savefig(dataset_path+Results_folder+"4.png")

# 5. Receptive Field Filter Weights
plt.figure(5)
filterWeights = weights[0][:,:,:,0]
numFrames = filterWeights.shape[2]
vmin = np.min(filterWeights)
vmax = np.max(filterWeights)
vabs = np.abs(filterWeights)
vabs_max = np.max(vabs)
for i in range(numFrames):
    plt.subplot(1,numFrames,i+1)
    plt.imshow(filterWeights[:,:,i],vmin=-vabs_max,vmax=+vabs_max)
plt.suptitle(' cell filter')
plt.savefig(dataset_path+Results_folder+"5.png")

# 6. Reconstructed Receptive Field Filter 
plt.figure(6)
reconFilter1=plotReconstruction(filterWeights,mapVals,Stride,Pool_Size,imSize[0])
plt.suptitle('Reconstruction of the linear filter')
plt.savefig(dataset_path+Results_folder+"6.png")

### PASS2 ###
# Initialize Filter Weights for Second Pass
Initial_Filter_Weights=[weights[0],weights[1]] # Receptive Field Estimates from Pass 1
Initial_exp=np.asarray([1]) # Intialize Power Law Exponet to 1

# Model for Pass 2 : Power Law Pass
model2=model_pass2(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize,Initial_Filter_Weights,Initial_exp)
model2.summary()
optimizerFunction = keras.optimizers.Adam(lr=0.001)
model2.compile(loss='mse', optimizer=optimizerFunction)    
earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
history=model2.fit(estSet,y_est, validation_data=(regSet,y_reg), epochs = 500,
              batch_size=750,callbacks=[earlyStop,mc],verbose=1)


# Calculate VAF
predicted_test_response = model2.predict(predSet)
predicted_test_response2 = predicted_test_response.reshape(-1)
respTest=y_pred.reshape(-1)
R=np.corrcoef(predicted_test_response2,respTest)
diag=R[0,1]
VAF_test2=diag*diag*100
print (VAF_test2)

# Pass 2 Trained Weights
weights2 = model2.get_weights()

# Plot Results
# 1. Learning Curve
plt.figure(7)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Learning Curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid()
plt.savefig(dataset_path+Results_folder+"7.png")

# 2. Actual/Predicted Response Data
plt.figure(8)
plt.plot(respTest[0:100],color='r',label='Actual')
plt.plot(predicted_test_response[0:100],color='b',label='Estimated')
plt.legend(loc='upper right')
plt.grid()
plt.title("Response Data")
plt.savefig(dataset_path+Results_folder+"8.png")

# 3. PReLU
plt.figure(9)
alpha1 = np.squeeze(weights2[2])
x = np.arange(-100,101)
y = np.arange(-100,101)
y[y<=0] = alpha1*y[y<=0] 
plt.plot(x,y)
plt.title('PReLU, alpha = {}'.format(np.round(alpha1,2)))
plt.savefig(dataset_path+Results_folder+"9.png")

# 4. Gaussian Map
plt.figure(10)
mapMean = weights2[3]
mapSigma = weights2[4]
mapVals = plotGaussMap(mapMean,mapSigma,downsampImageSize)
plt.title('Gaussian Map')
plt.savefig(dataset_path+Results_folder+"10.png")

# 5. Receptive Field Filter Weights
plt.figure(11)
filterWeights = weights2[0][:,:,:,0]
numFrames = filterWeights.shape[2]
vmin = np.min(filterWeights)
vmax = np.max(filterWeights)
vabs = np.abs(filterWeights)
vabs_max = np.max(vabs)
for i in range(numFrames):
    plt.subplot(1,numFrames,i+1)
    plt.imshow(filterWeights[:,:,i],vmin=-vabs_max,vmax=+vabs_max)
plt.suptitle(' cell filter')
plt.savefig(dataset_path+Results_folder+"11.png")

# 6. Reconstructed Receptive Field Filter 
plt.figure(12)
reconFilter2=plotReconstruction(filterWeights,mapVals,Stride,Pool_Size,imSize[0])
plt.suptitle('Reconstruction of the linear filter')
plt.savefig(dataset_path+Results_folder+"12.png")

# Save results in a .mat file
scipy.io.savemat(dataset_path+Results_folder+'Results_withoutBS.mat', {'Filter_Size':Filter_Size,
                                                                      'VAF_pass1':VAF_test1,'VAF_pass2':VAF_test2, 
                                                                      'weights_pass1': weights,'weights_pass2': weights2, 
                                                                      'Final_Rf_Construst_pass1': reconFilter1, 'Final_Rf_Construst_pass2': reconFilter2,
                                                                      'Predicted_response_pass1':predicted_test_response1, 'Predicted_response_pass2':predicted_test_response2})


