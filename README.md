# SystemIdentification_withCorticalStates

# Introduction

This repository contains Python code for a working example of our receptive field estimation method incorporating cortical state effects to account for trial-to-trial response variabilities, using a simple convolutional neural network approach. The method is described in detail, with results on many cortical neurons, in our paper currently submitted for publication

This repository contains code to run system identification with 2 example neurons used in the publication. When running the codes, make sure to change the folder names according to your setup.


# Requirements

Download LFP data from ... and add it to the '/Example_Neurons/LFPdata' folder


Python 3., scipy, numpy, matplotlib tensorflow 2.0, Keras. Check the environment.yml file for detailed list of dependencies. If you have an NVIDIA card, the libraries CUDA and CUDNN will be useful. https://developer.nvidia.com/cuda-zone https://developer.nvidia.com/cudnn
