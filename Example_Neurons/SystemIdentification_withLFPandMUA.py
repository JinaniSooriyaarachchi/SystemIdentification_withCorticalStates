#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:42:31 2022

@author: jinani

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
from k_arrangeLFPandMUA import arrange_LFP, arrange_MUA
from k_functions import conv_output_length,plotGaussMap,plotReconstruction
from k_model_withLFPandMUA import model_pass1, model_pass2

########################################
# INPUT DETAILS HERE !

dataset_name='H6903.013_1_Ch16' #'H6903.013_1_Ch16' for neuron1 and 'H6903.013_1_Ch28' for neuron2

#neuron1
Stimuli_dataset = sio.loadmat('/home/jinani/Desktop/Paper1_github/Example_Neurons/StimuliData/StimuliData_H6903.013_1_Ch16.mat') # neuron1 
##neuron2
#Stimuli_dataset = sio.loadmat('/home/jinani/Desktop/Paper1_github/Example_Neurons/StimuliData/StimuliData_H6903.013_1_Ch28.mat') # neuron2 

Channel=16    #16 for neuron1 and 28 for neuron2   
       
Results_folder='Results_withLFPandMUA/' 

LFPpath='/home/jinani/Desktop/Paper1_github/Example_Neurons/LFPdata/'  
MUApath='/home/jinani/Desktop/Paper1_github/Example_Neurons/MUAdata/MUAdata_H6903.013_1_Ch28/'

start_ch=15 #same for neuron1 and neuron2   
end_ch=32  #same for neuron1 and neuron2 

########################################

dataset_path='/home/jinani/Desktop/Paper1_github/Example_Neurons/'+dataset_name+'/'

# create results folder
dir = dataset_path+Results_folder
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)


num_timelags=7
Stride=1
Filter_Size=15
Pool_Size=1

# Arrange stimuli into time lags
estSet,regSet,predSet,imSize=arrange_stimuli(Stimuli_dataset,num_timelags)


# Arrange the Response Dataset
y_est,y_reg,y_pred=arrange_responses(dataset_path,dataset_name)

# Arrange the LFP dataset
estLFP, regLFP, predLFP = arrange_LFP(LFPpath, Channel, start_ch, end_ch)
lfp_length=(end_ch-start_ch)*150

# Arrange the MUA Dataset
estMUA, regMUA, predMUA = arrange_MUA (MUApath,Channel)
mua_length=2*150 # only above and below channel

# Calculate Shape of the Main Input and Intermediate Layer Inputs 
Input_Shape=estSet.shape[1:]   
numRows =  Input_Shape[0]  
numCols =  Input_Shape[1]  
assert numRows == numCols
numFrames = Input_Shape[2] 

convImageSize = conv_output_length(numRows,Filter_Size,'valid',Stride) # Input to Conv2D Layer
downsampImageSize = conv_output_length(convImageSize,Pool_Size,'valid',Pool_Size) # Input to Gaussian Map Layer


TrainedResults=[]
VAFs=[]
lfp_l2_values=[0.1, 0.25, 0.5,0.75,1] 
mua_l2_values=[0.1,0.5,1.0,1.5]

count=0
for lfp_l2_value in lfp_l2_values:
    for mua_l2_value in mua_l2_values:
        
        # Model for Pass 1 : Filter Estimate Pass
        model=model_pass1(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize, 
                lfp_length, lfp_l2_value, mua_length, mua_l2_value)
        
        model.summary()
        
        optimizerFunction = keras.optimizers.Adam(lr=0.001)
        model.compile(loss='mse', optimizer=optimizerFunction)    
        earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')      
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)    
        history=model.fit([estSet,estLFP,estMUA],y_est, validation_data=([regSet,regLFP,regMUA],y_reg), epochs = 500,
                      batch_size=750,callbacks=[earlyStop,mc],verbose=1)
                
        # Pass 1 Trained Weights
        weights = model.get_weights()

        # Initialize Filter Weights for Second Pass
        Initial_Filter_Weights=[weights[0],weights[1]] # Receptive Field Estimates from Pass 1
        Initial_exp=np.asarray([1])  # Intialize Power Law Exponet to 1
        
        # Model for Pass 2 : Power Law Pass
        model2=model_pass2(Input_Shape,Filter_Size,Stride,Pool_Size,downsampImageSize,Initial_Filter_Weights,
                Initial_exp, lfp_length, lfp_l2_value,mua_length, mua_l2_value) 
        
        model2.summary()
        
        optimizerFunction = keras.optimizers.Adam(lr=0.001)
        model2.compile(loss='mse', optimizer=optimizerFunction)    
        earlyStop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')        
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
        history=model2.fit([estSet,estLFP,estMUA],y_est, validation_data=([regSet,regLFP,regMUA],y_reg), epochs = 500,
                      batch_size=750,callbacks=[earlyStop,mc],verbose=1)
        
        # Calculate VAF
        predicted_test_response = model2.predict([predSet,predLFP,predMUA])
        predicted_test_response = predicted_test_response.reshape(-1)
        respTest=y_pred.reshape(-1)
        R=np.corrcoef(predicted_test_response,respTest)
        diag=R[0,1]
        VAF_test=diag*diag*100
        print (VAF_test)
        VAFs.append(VAF_test)
        
        
        # Pass 2 Trained Weights
        weights2 = model2.get_weights()
        
        # Plot Results
        # 1. Learning Curve
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Learning Curve')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.grid()
        plt.savefig(dataset_path+Results_folder+str(count*3+1)+".png")
        plt.show()
        
        # 2. Actual/Predicted Response Data
        plt.figure()
        plt.plot(respTest[0:100],color='r',label='Actual')
        plt.plot(predicted_test_response[0:100],color='b',label='Estimated')
        plt.legend(loc='upper right')
        plt.grid()
        plt.title("Response Data")
        #plt.savefig(dataset_path+Results_folder+str(count*8+2)+".png")
        
        # 3. PReLU
        plt.figure()
        alpha1 = np.squeeze(weights2[2])
        x = np.arange(-100,101)
        y = np.arange(-100,101)
        y[y<=0] = alpha1*y[y<=0] 
        plt.plot(x,y)
        plt.title('PReLU, alpha = {}'.format(np.round(alpha1,2)))
        #plt.savefig(dataset_path+Results_folder+str(count*8+3)+".png")
        
        # 4. Gaussian Map
        plt.figure()
        mapMean = weights2[3]
        mapSigma = weights2[4]
        mapVals = plotGaussMap(mapMean,mapSigma,downsampImageSize)
        plt.title('Gaussian Map')
        #plt.savefig(dataset_path+Results_folder+str(count*8+4)+".png")
        
        # 5. Receptive Field Filter Weights
        plt.figure()
        filterWeights = weights[0][:,:,:,0]
        numFrames = filterWeights.shape[2]
        vmin = np.min(filterWeights)
        vmax = np.max(filterWeights)
        vabs = np.abs(filterWeights)
        vabs_max = np.max(vabs)
        for i in range(numFrames):
            plt.subplot(1,numFrames,i+1)
            plt.imshow(filterWeights[:,:,i],vmin=-vabs_max,vmax=+vabs_max)
        plt.suptitle('cell filter')
        plt.savefig(dataset_path+Results_folder+str(count*3+2)+".png")
        
        # 6. Reconstructed Receptive Field Filter 
        plt.figure()
        reconFilter=plotReconstruction(filterWeights,mapVals,Stride,Pool_Size,imSize[0])
        plt.suptitle('Reconstruction of the linear filter')
        plt.savefig(dataset_path+Results_folder+str(count*3+3)+".png")
        
        # 7. LFP temporal filters
        TfEstimate=weights2[5]
        plt.figure()
        plt.plot(TfEstimate)
        plt.grid()
        plt.title("Temporal Filter_Estimated-lfp")
        #plt.savefig(dataset_path+Results_folder+str(count*8+7)+".png")
        
        # 8. MUA temporal filters
        TfEstimate=weights2[7]
        plt.figure()
        plt.plot(TfEstimate)
        plt.grid()
        plt.title("Temporal Filter_Estimated-mua")
        #plt.savefig(dataset_path+Results_folder+str(count*8+8)+".png")
    
        count+=1
        
        TrainedResults.append({'weights_pass1': weights,'weights_pass2': weights2, 
                                                'Final_Rf_Construst': reconFilter,'VAF':VAF_test, 
                                                'Predicted_response':predicted_test_response})
    

# Save results in a .mat file
scipy.io.savemat(dataset_path+Results_folder+'Results_WithLFPMUAMethod.mat', {'TrainedResults': TrainedResults,'VAFs': VAFs, 
                                                'LFP_L2_values':lfp_l2_values,'MUA_L2_values':mua_l2_values})






