#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:43:34 2021

@author: dani
"""
# Neural networks libraries
import torch
from torch import nn,optim

# Numpy libraries for array and matrix operations
import numpy as np
from numpy import diag
from numpy import zeros

# Matplotlib for figure and plot building
import matplotlib.pyplot as plt

# System, path and svd information
import os
import random

import glob
from scipy.linalg import svd

# Call for all the subfunctions needed from the rest of the scripts
import NNmodules
 
def DataDoF(path):
    # Create a list with both old & new data sampling
    DirList = glob.glob("../caseDatabase/*")
    print("The amount of available cases is '% s'" % len(DirList))
    BCList = ['inlet_U.xy', 'outlet_U.xy','inlet_T_p.xy', 'outlet_T_p.xy']
    # Call for the amplitude and wavelength data from each and every case in existance
    Amp = []
    WaveL = []
    for Diri in DirList:
        AWfile = open(Diri+"/AmpWaveL.txt")
        AWcont = AWfile.readlines()
        Ampin = AWcont[0]
        WaveLin = AWcont[1]
        AWfile.close()
        Amp.append(Ampin)
        WaveL.append(WaveLin)
    Amp = np.float32(Amp)
    WaveL = np.float32(WaveL)
    x = np.empty([len(Amp),2])
    xin = np.empty([len(Amp),2])
    # Standardize the data for each feature Amp and WaveL
    xsd0 = np.std(Amp)
    xsd1 = np.std(WaveL)
    xmean0 = np.mean(Amp)
    xmean1 = np.mean(WaveL)
    # Build the matrix of features, in this case its 2 features with Amp x WaveL snapshots
    for xi in range(0,len(x)):
        xin[xi,0] = (Amp[xi]-xmean0)/xsd0
        xin[xi,1] = (WaveL[xi]-xmean1)/xsd1
    x[:,0] = xin[:,0]
    x[:,1] = xin[:,1]
    
    # Call for each case data, these files start as lists to append
    inlet = []
    outlet = []
    wall = []
    invelocity = []
    outvelocity = []
    # wallvelocity = []
    # Each case is loaded separately within the loop to append to the total list
    print("Loading data to feed the NNs and RBF...")
    for p in DirList:
        for bc in BCList:
            # Built the path to call for each case
            filepath = p+'/postProcessing/sample/Helium/1000/'+bc
            # Store the current case, BC & variable data in a numpy array,
            # then transform said array into a tensor for the NN
            Varin = torch.from_numpy(np.loadtxt(filepath))
            # The list.append() requires the data type "float", so the tensor is
            # transformed to float
            Varin = Varin.float()
            # Now append the float tensor into the list of each variable & BC
            if bc == 'inlet_U.xy':
                invelocity.append(Varin)
            elif bc == 'outlet_U.xy':
                outvelocity.append(Varin)
            # elif bc == 'wall_U.xy':
            #     wallvelocity.append(Varin)
            elif bc == 'inlet_T_p.xy':
                inlet.append(Varin)
            elif bc == 'outlet_T_p.xy':
                outlet.append(Varin)
            # elif bc == 'wall_T_p.xy':
            #     wall.append(Varin)
    # Transform the lists into a matrix of tensors of shape U = snapshots x DoF x coordinate
    # while TP = snapshots x DoF x coor,T,P location in file
    sui = torch.stack(invelocity)
    suo = torch.stack(outvelocity)
    # suw = torch.stack(wallvelocity)
    sin = torch.stack(inlet)
    sout = torch.stack(outlet)
    # swall = torch.stack(wall)
    
    print("Standardizing data for the NNs and RBF...")
    # Built the empty tensor for the standardized data to be built from the variables
    # by taking the standard deviation and mean values from each tensor
    stanui = torch.empty(sui[:,:,1].shape)
    stanuo = torch.empty(suo[:,:,1].shape)
    # stanuw = torch.empty(suw[:,:,1].shape)
    stanTi = torch.empty(sin[:,:,1].shape)
    stanTo = torch.empty(sout[:,:,1].shape)
    # stanTw = torch.empty(swall[:,:,1].shape)
    stanpi = torch.empty(sin[:,:,2].shape)
    stanpo = torch.empty(sout[:,:,2].shape)
    # stanpw = torch.empty(swall[:,:,2].shape)
    # BCtensorList = [sui[:,:,1], suo[:,:,1], suw[:,:,1], sin[:,:,1], sout[:,:,1], swall[:,:,1], sin[:,:,2], sout[:,:,2], swall[:,:,2]]
    BCtensorList = [sui[:,:,1], suo[:,:,1], sin[:,:,1], sout[:,:,1], sin[:,:,2], sout[:,:,2]]
    index = 0
    for bct in BCtensorList:
        bctensor = bct
        stdTensor = np.std(bctensor.detach().numpy())
        meanTensor = np.mean(bctensor.detach().numpy())
        # Store the standard deviation and mean values for each BC to calculate
        # later the real variable values
        stdfile = open("stdData.txt")
        stdstring = stdfile.readlines()
        stdfile.close()
        stdstring[index] = str(stdTensor)
        stdfile = open("stdData.txt", "w")
        newstdcont = "".join(stdstring)
        stdfile.write(newstdcont)
        stdfile.close()
        meanfile = open("meanData.txt")
        meanstring = meanfile.readlines()
        meanfile.close()
        meanstring[index] = str(meanTensor)
        meanfile = open("meanData.txt", "w")
        newmeancont = "".join(meanstring)
        meanfile.write(newmeancont)
        meanfile.close()
        for i in range(0,len(DirList),1):
            if index == 0:
                stanui[i,:] = (sui[i,:,1]-meanTensor)/stdTensor
            elif index == 1:
                stanuo[i,:] = (suo[i,:,1]-meanTensor)/stdTensor
            elif index == 2:
                stanTi[i,:] = (sin[i,:,1]-meanTensor)/stdTensor
            elif index == 3:
                stanTo[i,:] = (sout[i,:,1]-meanTensor)/stdTensor
            elif index == 4:
                stanpi[i,:] = (sin[i,:,2]-meanTensor)/stdTensor
            elif index == 5:
                stanpo[i,:] = (sout[i,:,2]-meanTensor)/stdTensor
                
                
        index = index + 1
    stanBCList = [stanui, stanuo, stanTi, stanTo, stanpi, stanpo] # boundary  
    stanBCListNames = ["stanui", "stanuo", "stanTi", "stanTo", "stanpi", "stanpo"] # boundary  
    # for strStanBC in stanBCListNames:
    #     ParentDir = '/media/dani/Data/Ubuntufiles/ProperThermoProp/SVDdata/'
    #     NewPath = os.path.join(ParentDir, strStanBC)
    #     os.mkdir(NewPath)
    return x, stanBCList, stanBCListNames
    
    
def DataSVD(x, stanBC, stanBCname):
    # define a matrix
    A = stanBC
    # Singular-value decomposition
    U, s, VT = svd(A)
    modes = 5
    # create m x n Sigma matrix
    Sigma = zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    if len(x) > len(stanBC[0,:]):
        Sigma[:A.shape[1], :A.shape[1]] = diag(s)
    else:
        Sigma[:A.shape[0], :A.shape[0]] = diag(s)
    # select the main modes for the Sigma to transform U into modes x snapshots
    n_elements = modes
    Sigma = Sigma[:, :n_elements]
    VT = VT[:n_elements, :]
    # New modes x snapshots reduced matrix
    Trin = U.dot(Sigma)
    # Save the VT reduced data to transform back the matrix into snapshots x DoF
    name = '../SVDdata/'+stanBCname+'/VT_Data.csv'
    np.savetxt(name, VT, delimiter=",")
    # Standardize the modes x snapshots matrix
    Tr = np.empty(Trin.shape)
    Trsd = np.std(Trin)
    Trmean = np.mean(Trin)
    for tri in range(0,len(Trin),1):
        for mj in range(0,modes,1):
            Tr[tri,mj] = (Trin[tri,mj]-Trmean)/Trsd
    
    dataset = stanBC
    batch_size = len(x[:,1]) * len(x[:,0]) -1
    validation_split = 0.35
    
    # Creating data indices for training and validation splits:
    bT = np.asarray(np.where(x[:,1] == min(x[:,1])))
    tT = np.asarray(np.where(x[:,1] == max(x[:,1])))
    bT = list(np.squeeze(bT, axis=(0,)))
    tT = list(np.squeeze(tT, axis=(0,)))
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    indices2 = indices
    for b in bT:
        indices2.remove(b)
    for t in tT:
        indices2.remove(t)
    
    # Don't use any of the upper and lower value limits for validation
    split = int(np.floor(validation_split * dataset_size))
    random.shuffle(indices2)
    train_indices, val_indices = indices2[split:], indices2[:split]
    for b in bT:
        train_indices.append(b)
    for t in tT:
        train_indices.append(t)
    
    # Create the validation and training datasets for the SVD, DoF and RBF
    x_train = torch.from_numpy(np.float32(x[train_indices]))
    x_valid = torch.from_numpy(np.float32(x[val_indices]))
    stanBC_train = stanBC[train_indices,:]
    stanBC_valid = stanBC[val_indices,:]
    Tr_train = torch.from_numpy(np.float32(Tr[train_indices,:]))
    Tr_valid = torch.from_numpy(np.float32(Tr[val_indices,:]))
    
    return modes, x_train, x_valid, stanBC_train, stanBC_valid, Tr_train, Tr_valid

def UpdateParametersSVD(stanBCname, epochs, modes, x_train, x_valid, Tr_train, Tr_valid):
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    hidden_sizes = 32
    D_in = 2
    D_out = modes
    
    # call for the SVD network module
    network = NNmodules.NNSVD(hidden_sizes, D_in, D_out)
    # Create the optimizer
    # Adagrad (not so good)
    # Adam (not so good)
    # AdamW
    # Adamax
    # ASGD (not so good)
    # RMSprop (not so good)
    # Rprop (best so far)
    # SGD (worst so far, requires lr = 0.003)
    # optimizer = optim.Rprop([
    #     {'params': network1.parameters()},
    #     {'params': network2.parameters()}])
    optimizer = optim.Rprop(network.parameters())
    
    # loss function is mean squared error
    loss_fn = torch.nn.MSELoss()
    
    stop_criteria = 0.0000000001
    LOSS = []
    LOSSval = []
    # Checkpoint name for the new neural network
    checkpoint_path='../SVDdata/'+stanBCname+'/SVD-outData-NN.pt'
    checkpoint={'epoch':None,'model_state_dict':None ,'optimizer_state_dict':None ,'loss': None, 'lossv': None}
    # Start the training feedforward/backpropagation loop
    for epoch in range(epochs):
        yhat1 = network(x_train)
        loss1 = loss_fn(yhat1, Tr_train)
        LOSS.append(loss1.item())
        v = network(x_valid)
        loss2v = loss_fn(v, Tr_valid)
        LOSSval.append(loss2v.item())
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        if LOSS[-1] < stop_criteria:
            print('SVD', stanBCname, ' NN minimal loss criteria has been reached within epoch', epoch)
            break
    # Save the checkpoint into a file to use in the future within the inverse problem
    checkpoint['epoch']=epochs
    checkpoint['network_state_dict']=network.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['loss']=LOSS
    checkpoint['lossv']=LOSSval
    torch.save(checkpoint, checkpoint_path)
    
    
def UpdateParametersDoF(stanBCname, epochs, x_train, x_valid, stanBC_train, stanBC_valid):
    hidden_sizes = 32
    D_in = 2
    D_out = len(stanBC_train[0,:])
    
    # call for the DoF network module
    network = NNmodules.NNDoF(hidden_sizes, D_in, D_out)
    # Create the optimizer
    # Adagrad (not so good)
    # Adam (not so good)
    # AdamW
    # Adamax
    # ASGD (not so good)
    # RMSprop (not so good)
    # Rprop (best so far)
    # SGD (worst so far, requires lr = 0.003)
    # optimizer = optim.Rprop([
    #     {'params': network1.parameters()},
    #     {'params': network2.parameters()}])
    optimizer = optim.Rprop(network.parameters())
    
    # loss function is mean squared error
    loss_fn = torch.nn.MSELoss()
    
    stop_criteria = 0.000000001
    LOSS = []
    LOSSval = []
    # Checkpoint name for the new neural network
    checkpoint_path='../SVDdata/'+stanBCname+'/DoF-outData-NN.pt'
    # Start the training feedforward/backpropagation loop
    checkpoint={'epoch':None,'model_state_dict':None ,'optimizer_state_dict':None ,'loss': None, 'lossv': None}
    # Decoupled NNs, 2 NN
    for epoch in range(epochs):
        yhat1 = network(x_train)
        loss1 = loss_fn(yhat1, stanBC_train)
        LOSS.append(loss1.item())
        v = network(x_valid)
        loss2v = loss_fn(v, stanBC_valid)
        LOSSval.append(loss2v.item())
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        if LOSS[-1] < stop_criteria:
            print('DoF', stanBCname, ' NN minimal loss criteria has been reached within epoch', epoch)
            break
    # Save the checkpoint into a file to use in the future within the inverse problem
    checkpoint['epoch']=epochs
    checkpoint['network_state_dict']=network.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['loss']=LOSS
    checkpoint['lossv']=LOSSval
    torch.save(checkpoint, checkpoint_path)
    
def UpdateParametersRBF(stanBCi, modes, x_train, x_valid, Tr_train, Tr_valid):
    D_out = modes
    NNmodules.PODRBF()
    