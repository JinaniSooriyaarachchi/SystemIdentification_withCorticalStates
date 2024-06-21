#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:42:31 2022

@author: jinani

"""
# Import Required Libraries
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 1. downsample: To downsample stimuli from (crop_window x crop_window) to (30 x 30)
def downsample(stimuli,kernel_size): # kernel_size = downsampling factor
    kernel=np.ones((kernel_size,kernel_size))*(1/(kernel_size**2))
    downsample_stimuli=None
    for i in range(np.shape(stimuli)[2]):
        array=stimuli[:,:,i]
        array_downsampled =convolve(array, kernel)[:array.shape[0]:kernel_size,:array.shape[1]:kernel_size]
        if i==0:
            downsample_stimuli=array_downsampled
        else:
            downsample_stimuli=np.dstack((downsample_stimuli,array_downsampled))
    return (downsample_stimuli)

# 2. buildCalcAndScale: To normalize the stimuli data
def buildCalcAndScale(stim):
    def standardize(X, given_axis , parameters):
        given_mean=parameters[0]
        given_std=parameters[1]
        Xnorm = X - given_mean
        Xnorm = Xnorm / given_std
        Xnorm = np.nan_to_num(Xnorm) # Replace NaN with zero and infinity with large finite number
        return Xnorm
    
    def calcStandardize(X,given_axis = None):
        Xmean = np.mean(X,axis=given_axis)
        Xstd = np.std(X,axis=given_axis)
        return Xmean,Xstd
    
    axis = 0    
    featureScalingParams = calcStandardize(stim,axis)    
    stim = standardize(stim,axis,featureScalingParams)
    
    return stim

# 3. dataDelay, dataDelayAsList, kConvNetStyle: To create time lagged data
def dataDelay(stim,trialSize,delay =[0]):
    ''' for every time point, new stim is the set of previous frames (delay ==0 is the current frame)
    Inputs: stim(m,n) array, m is the features, m is the examples
            trialSize, size of each trial, if you need a frame before the trial begins, it will be a zero-filled frame
            delay, array of delays to use. each delay corresponds to a previous input, ex delay = range(8), use all up to 7 preceding frames, 
            if delay = [0], new stim will be the same 
            if delay = [2], new stim will use only the stimulus from 2 frames ago'''            
    stimSize = np.size(stim,axis=0)  #7500
    splitIndices = np.arange(0,stimSize-trialSize,trialSize)+trialSize
    splitList = np.split(stim,splitIndices,axis=0)
    #fill stim with zeros, to prepare for adding delays
    stim = np.zeros((stimSize,np.size(delay)*np.size(stim,axis=1)))
    for trialNum in range(len(splitList)):
        trial = splitList[trialNum]
        for frameNum in range(np.size(trial,axis=0)):
            stimFrame = []
            for k in delay:
                delayNum  = frameNum -k
                
                if delayNum < 0:
                    delayFrame = np.zeros(np.shape(trial[delayNum,:]))
                else:
                    delayFrame = trial[delayNum,:]  
                        
                stimFrame = np.concatenate((stimFrame,delayFrame))
            stim[frameNum+trialNum*trialSize,:] = stimFrame   
    return stim    
def dataDelayAsList(stim,numFrames):
    stim = np.split(stim,numFrames,axis= 1)
    stim = np.dstack(stim)
    stim = np.swapaxes(stim,1,2)
    return stim
def kConvNetStyle(stim,Frames):
    ''' with standard a_movieClip, the movies are 375 frames long'''    
    trialSize = 375
    X = dataDelay(stim,trialSize,Frames)
    X = dataDelayAsList(X,len(Frames))
    #convert to (samples,frames,rows,cols)
    X = X.reshape(X.shape[0],X.shape[1],np.int(np.sqrt(X.shape[2])),np.int(np.sqrt(X.shape[2])))
    X=np.swapaxes(X,1,3)
    X=np.swapaxes(X,1,2)
    return X

# 4. conv_output_length: To calculate the output of convolution layer
def conv_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)  
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1  
    return (output_length + stride - 1) // stride  

# 5. plotGaussMap: To plot gaussian map
def plotGaussMap(mean,sigma,mapSize):
    sigmaVector = np.asarray([[sigma[0],sigma[1]],[sigma[1],sigma[2]]])
    
    meanVector = mean*mapSize
    sigmaVector = sigmaVector*mapSize
        
    x,y = np.meshgrid(range(mapSize),range(mapSize))
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    mvn = multivariate_normal(meanVector,sigmaVector) 
    myMap = mvn.pdf(pos)
    plt.imshow(myMap)
    return myMap

# 6. plotReconstruction, plotFilter: To plot receptive field reconstruction
def plotFilter(filterWeights):
    numFrames = filterWeights.shape[2]
    #vmin = np.min(filterWeights)
    #vmax = np.max(filterWeights)
    vabs = np.abs(filterWeights)
    vabs_max = np.max(vabs)
    for i in range(numFrames):
        plt.subplot(1,numFrames,i+1)
        plt.imshow(filterWeights[:,:,i],vmin=-vabs_max,vmax=+vabs_max)
    return
def plotReconstruction(filterWeights,mapWeights,stride,poolSize,fullSize):
    mapSize = np.shape(mapWeights)[0]
    filterSize = np.shape(filterWeights)[0]
    numLags = np.shape(filterWeights)[2]
    unPoolFilter = np.zeros((mapSize*poolSize*stride,mapSize*poolSize*stride))
    reconFilter = np.zeros((fullSize,fullSize,numLags))
    for map_x_idx in range(mapSize):
        for map_y_idx in range(mapSize):

            unPoolFilter[(map_y_idx)*poolSize*stride:(map_y_idx+1)*poolSize*stride,(map_x_idx)*poolSize*stride:(map_x_idx+1)*poolSize*stride] = (
            unPoolFilter[(map_y_idx)*poolSize*stride:(map_y_idx+1)*poolSize*stride,(map_x_idx)*poolSize*stride:(map_x_idx+1)*poolSize*stride] +mapWeights[map_y_idx,map_x_idx])
    
    for lag in range(numLags):
        for x_idx in range(mapSize*poolSize*stride):
            for y_idx in range(mapSize*poolSize*stride):
                reconFilter[x_idx:x_idx+filterSize,y_idx:y_idx+filterSize,lag] = (
                        reconFilter[x_idx:x_idx+filterSize,y_idx:y_idx+filterSize,lag] +unPoolFilter[x_idx,y_idx]*filterWeights[:,:,lag])
    plotFilter(reconFilter)
    
    return reconFilter