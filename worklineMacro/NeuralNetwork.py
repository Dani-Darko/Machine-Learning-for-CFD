"""
1. NeuralNetwork.py creates the dataset for training from the /caseDatabase
    folder and normaizes the data within the def DataDoF(), gets the modes
    from the dataset to decrease the dimensions of the output tensor within
    def DataSVD(), creates the def for the DoF(), SVD(), QoI(), RBF() and
    Kriging() update training and calls for the Neural network class within
    where is required.

@author: Daniela Segura
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
import sklearn.preprocessing as pp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern, RationalQuadratic, ExpSineSquared

from joblib import dump

# Call for all the subfunctions needed from the rest of the scripts
import NNmodules

hidden_sizes = 32
 
def DataDoF(path):
    # Create a list with both old & new data sampling
    DirList1 = glob.glob("../caseDatabase/*")
    DirList = []
    for Dir1 in DirList1:
        if os.path.exists(Dir1 + '/postProcessing'):
            DirList.append(Dir1)
    print("The amount of available cases is '% s'" % len(DirList))
    BCList = ['inlet2_U.xy', 'outlet4_U.xy','inlet2_T_p.xy', 'outlet4_T_p.xy']
    # Call for the amplitude and wavelength data from each and every case in existance
    AW = []
    for Diri in DirList:
        AWfile = np.genfromtxt(Diri+'/AmpWaveLNN.txt', delimiter=' ')
        AW.append(AWfile)
    print("The amount of available cases is '% s'" % len(DirList))
    feat = len(AWfile)
    AW = np.asarray(AW)
    x = np.empty([len(AW),feat])
    xin = np.empty([len(AW),feat])
    # Standardize the data for each feature Amp and WaveL
    xmean = np.empty(feat)
    xsd = np.empty(feat)
    # Build the matrix of features, in this case its 2 features with Amp x WaveL snapshots
    for xj in range(feat):
        xmean[xj] = np.mean(AW[:,xj])
        xsd[xj] = np.std(AW[:,xj])
        for xi in range(len(AW)):
            xin[xi,xj] = (AW[xi,xj]-xmean[xj])/xsd[xj]
    x = xin
    
    # Call for each case data, these files start as lists to append
    inlet = []
    outlet = []
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
            if bc == 'inlet2_U.xy':
                invelocity.append(Varin)
            elif bc == 'outlet4_U.xy':
                outvelocity.append(Varin)
            # elif bc == 'wall_U.xy':
            #     wallvelocity.append(Varin)
            elif bc == 'inlet2_T_p.xy':
                inlet.append(Varin)
            elif bc == 'outlet4_T_p.xy':
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
    # BCtensorList = [sui[:,:,1], suo[:,:,1], suw[:,:,1], sin[:,:,1], sout[:,:,1], swall[:,:,1], sin[:,:,2], sout[:,:,2], swall[:,:,2]]
    BCtensorList = [sui[:,:,1], suo[:,:,1], sin[:,:,1], sout[:,:,1], sin[:,:,2], sout[:,:,2]]
    stdList = np.empty([len(BCtensorList)])
    meanList = np.empty([len(BCtensorList)])
    stanBCList = []
    for bct in range(len(BCtensorList)):
        # Built the empty tensor for the standardized data to be built from the variables
        # by taking the standard deviation and mean values from each tensor
        stan = torch.empty(BCtensorList[bct].shape)
        stdList[bct] = np.std(BCtensorList[bct].detach().numpy())
        meanList[bct] = np.mean(BCtensorList[bct].detach().numpy())
        if stdList[bct] == 0.0:
            stdList[bct] = 1.0
            meanList[bct] = 0.0
        # Store the standard deviation and mean values for each BC to calculate
        # later the real variable values
        for i in range(0,len(BCtensorList[bct]),1):
            stan[i,:] = (BCtensorList[bct][i,:]-meanList[bct])/stdList[bct]
        stanBCList.append(stan)
    if os.path.exists("stdData.txt"):
        os.remove("stdData.txt")
    stdfile = open("stdData.txt", "a")
    for line in range(len(stdList)):
        stdfile.write(str(stdList[line]))
        stdfile.write("\n")
    stdfile.close()
    if os.path.exists("meanData.txt"):
        os.remove("meanData.txt")
    meanfile = open("meanData.txt", "a")
    for line in range(len(meanList)):
        meanfile.write(str(meanList[line]))
        meanfile.write("\n")
    meanfile.close()
    # stanBCList = [stanui, stanuo, stanTi, stanTo, stanpi, stanpo] # boundary  
    stanBCListNames = ["stanui", "stanuo", "stanTi", "stanTo", "stanpi", "stanpo"] # boundary  
    return x, stanBCList, stanBCListNames, BCtensorList, xsd, xmean, feat
    
    
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
    Trsd = np.std(Trin)
    Trmean = np.mean(Trin)
    if os.path.exists('../SVDdata/'+stanBCname+'/TrstdmeanData.txt'):
        os.remove('../SVDdata/'+stanBCname+'/TrstdmeanData.txt')
    stdfile = open('../SVDdata/'+stanBCname+'/TrstdmeanData.txt', "a")
    stdfile.write(str(Trsd))
    stdfile.write("\n")
    stdfile.write(str(Trmean))
    stdfile.close()
    # Save the VT reduced data to transform back the matrix into snapshots x DoF
    name = '../SVDdata/'+stanBCname+'/VT_Data.csv'
    np.savetxt(name, VT, delimiter=",")
    # Standardize the modes x snapshots matrix
    Tr = np.empty(Trin.shape)
    for mj in range(0,modes,1):
        for tri in range(0,len(Trin),1):
            Tr[tri,mj] = (Trin[tri,mj]-Trmean)/Trsd
    
    dataset = stanBC
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
    x_train = torch.from_numpy(np.float32(x[train_indices,:]))
    # x_train = torch.from_numpy(np.float32(x[:,:]))
    x_valid = torch.from_numpy(np.float32(x[val_indices]))
    stanBC_train = stanBC[train_indices,:]
    # stanBC_train = stanBC[:,:]
    stanBC_valid = stanBC[val_indices,:]
    Tr_train = torch.from_numpy(np.float32(Tr[train_indices,:]))
    # Tr_train = torch.from_numpy(np.float32(Tr[:,:]))
    Tr_valid = torch.from_numpy(np.float32(Tr[val_indices,:]))
    
    return modes, x_train, x_valid, stanBC_train, stanBC_valid, Tr_train, Tr_valid

def UpdateParametersSVD(D_in, stanBCname, epochs, modes, x_train, x_valid, Tr_train, Tr_valid):
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
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
    checkpoint['model_state_dict']=network.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['loss']=LOSS
    checkpoint['lossv']=LOSSval
    torch.save(checkpoint, checkpoint_path)
    
    print("Calculate the L2 norm error from each training and validation sets case")
    L2t = []
    L2v = []
    for i in range(len(Tr_train[:,0])):
        er = np.sqrt(sum((yhat1[i,:] - Tr_train[i,:])**2))
        l2errnorm = er * np.sqrt(0.2*3.0)/np.sqrt(len(yhat1[i,:]))
        L2t.append(l2errnorm)
    for i in range(len(Tr_valid[:,0])):
        er = np.sqrt(sum((v[i,:] - Tr_train[i,:])**2))
        l2errnorm = er * np.sqrt(0.2*3.0)/np.sqrt(len(v[i,:]))
        L2v.append(l2errnorm)
    if os.path.exists('../SVDdata/'+stanBCname+"L2tsvd.txt"):
        os.remove('../SVDdata/'+stanBCname+"L2tsvd.txt")
    stdfile = open('../SVDdata/'+stanBCname+"L2tsvd.txt", "a")
    for line in range(len(L2t)):
        stdfile.write(str(L2t[line]))
        stdfile.write("\n")
    stdfile.close()
    if os.path.exists('../SVDdata/'+stanBCname+"L2vsvd.txt"):
        os.remove('../SVDdata/'+stanBCname+"L2vsvd.txt")
    meanfile = open('../SVDdata/'+stanBCname+"L2vsvd.txt", "a")
    for line in range(len(L2v)):
        meanfile.write(str(L2v[line]))
        meanfile.write("\n")
    meanfile.close()
    
    
    # fig_pathLOSSSVD = '../SVDdata/'+stanBCname+'/SVDLossPerformance.eps'
    # plt.plot(LOSS)
    # plt.plot(LOSSval)
    # plt.legend(['Training Temp', 'Validation Temp'], loc='upper right', fontsize=12)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.grid(color="0.8", linewidth=0.5) 
    # plt.xlabel("Epoch", size=14)
    # plt.ylabel("Cost/total loss", size=14)
    # plt.yscale('log')
    # plt.savefig(fig_pathLOSSSVD, bbox_inches='tight', format='eps')
    # plt.close()
    
    
def UpdateParametersDoF(D_in, stanBCname, epochs, x_train, x_valid, stanBC_train, stanBC_valid):
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
    checkpoint['model_state_dict']=network.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['loss']=LOSS
    checkpoint['lossv']=LOSSval
    torch.save(checkpoint, checkpoint_path)
    
    print("Calculate the L2 norm error from each training and validation sets case")
    L2t = []
    L2v = []
    for i in range(len(stanBC_train[:,0])):
        er = np.sqrt(sum((yhat1[i,:] - stanBC_train[i,:])**2))
        l2errnorm = er * np.sqrt(0.2*3.0)/np.sqrt(len(yhat1[i,:]))
        L2t.append(l2errnorm)
    for i in range(len(stanBC_valid[:,0])):
        er = np.sqrt(sum((v[i,:] - stanBC_train[i,:])**2))
        l2errnorm = er * np.sqrt(0.2*3.0)/np.sqrt(len(v[i,:]))
        L2v.append(l2errnorm)
    if os.path.exists('../SVDdata/'+stanBCname+"L2tdof.txt"):
        os.remove('../SVDdata/'+stanBCname+"L2tdof.txt")
    stdfile = open('../SVDdata/'+stanBCname+"L2tdof.txt", "a")
    for line in range(len(L2t)):
        stdfile.write(str(L2t[line]))
        stdfile.write("\n")
    stdfile.close()
    if os.path.exists('../SVDdata/'+stanBCname+"L2vdof.txt"):
        os.remove('../SVDdata/'+stanBCname+"L2vdof.txt")
    meanfile = open('../SVDdata/'+stanBCname+"L2vdof.txt", "a")
    for line in range(len(L2v)):
        meanfile.write(str(L2v[line]))
        meanfile.write("\n")
    meanfile.close()
    
    # fig_pathLOSSSVD = '../SVDdata/'+stanBCname+'/DoFLossPerformance.eps'
    # plt.plot(LOSS)
    # plt.plot(LOSSval)
    # plt.legend(['Training Temp', 'Validation Temp'], loc='upper right', fontsize=12)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.grid(color="0.8", linewidth=0.5) 
    # plt.xlabel("Epoch", size=14)
    # plt.ylabel("Cost/total loss", size=14)
    # plt.yscale('log')
    # plt.savefig(fig_pathLOSSSVD, bbox_inches='tight', format='eps')
    # plt.close()

def UpdateParametersQoI(D_in, Q, epochs, x, name):
    Qstd = np.std(Q)
    Qmean = np.mean(Q)
    if os.path.exists('../SVDdata/QoI/Q-'+name+'-stdmeanData.txt'):
        os.remove('../SVDdata/QoI/Q-'+name+'-stdmeanData.txt')
    stdfile = open('../SVDdata/QoI/Q-'+name+'-stdmeanData.txt', "a")
    stdfile.write(str(Qstd))
    stdfile.write("\n")
    stdfile.write(str(Qmean))
    stdfile.close()
    # Standardize the modes x snapshots matrix
    Qout = np.empty(Q.shape)
    for i in range(0,len(Q),1):
        Qout[i] = (Q[i]-Qmean)/Qstd
    
    dataset = Qout
    validation_split = 0.35
    # Creating data indices for training and validation splits:
    # bT = np.asarray(np.where(Q == min(Q)))
    # tT = np.asarray(np.where(Q == max(Q)))
    # bT = list(np.squeeze(bT, axis=(0,)))
    # tT = list(np.squeeze(tT, axis=(0,)))
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    indices2 = indices
    # for b in bT:
    #     indices2.remove(b)
    # for t in tT:
    #     indices2.remove(t)
    
    # Don't use any of the upper and lower value limits for validation
    split = int(np.floor(validation_split * dataset_size))
    random.shuffle(indices2)
    train_indices, val_indices = indices2[split:], indices2[:split]
    # for b in bT:
    #     train_indices.append(b)
    # for t in tT:
    #     train_indices.append(t)
    
    # Create the validation and training datasets for the SVD, DoF and RBF
    # x_train = torch.from_numpy(np.float32(x[train_indices,:]))
    x_train = torch.from_numpy(np.float32(x[:,:]))
    x_valid = torch.from_numpy(np.float32(x[val_indices]))
    # Q_train = torch.unsqueeze(torch.from_numpy(np.float32(Qout[train_indices])),1)
    Q_train = torch.unsqueeze(torch.from_numpy(np.float32(Qout[:])),1)
    Q_valid = torch.unsqueeze(torch.from_numpy(np.float32(Qout[val_indices])),1)
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    D_out = 1
    
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
    checkpoint_path='../SVDdata/QoI/QoI-'+name+'-NN.pt'
    checkpoint={'epoch':None,'model_state_dict':None ,'optimizer_state_dict':None ,'loss': None, 'lossv': None}
    # Start the training feedforward/backpropagation loop
    for epoch in range(epochs):
        yhat1 = network(x_train)
        loss1 = loss_fn(yhat1, Q_train)
        LOSS.append(loss1.item())
        v = network(x_valid)
        loss2v = loss_fn(v, Q_valid)
        LOSSval.append(loss2v.item())
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        if LOSS[-1] < stop_criteria:
            print('SVD', name, ' NN minimal loss criteria has been reached within epoch', epoch)
            break
    # Save the checkpoint into a file to use in the future within the inverse problem
    checkpoint['epoch']=epochs
    checkpoint['model_state_dict']=network.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['loss']=LOSS
    checkpoint['lossv']=LOSSval
    torch.save(checkpoint, checkpoint_path)
    
    print("Calculate the L2 norm error from each training and validation sets case")
    L2t = []
    L2v = []
    for i in range(len(Q_train[:,0])):
        er = np.sqrt(sum((yhat1[i,:] - Q_train[i,:])**2))
        l2errnorm = er * np.sqrt(0.2*3.0)/np.sqrt(len(yhat1[i,:]))
        L2t.append(l2errnorm)
    for i in range(len(Q_valid[:,0])):
        er = np.sqrt(sum((v[i,:] - Q_train[i,:])**2))
        l2errnorm = er * np.sqrt(0.2*3.0)/np.sqrt(len(v[i,:]))
        L2v.append(l2errnorm)
    if os.path.exists('../SVDdata/QoI/'+name+"L2tdof.txt"):
        os.remove('../SVDdata/QoI/'+name+"L2tdof.txt")
    stdfile = open('../SVDdata/QoI/'+name+"L2tdof.txt", "a")
    for line in range(len(L2t)):
        stdfile.write(str(L2t[line]))
        stdfile.write("\n")
    stdfile.close()
    if os.path.exists('../SVDdata/QoI/'+name+"L2vdof.txt"):
        os.remove('../SVDdata/QoI/'+name+"L2vdof.txt")
    meanfile = open('../SVDdata/QoI/'+name+"L2vdof.txt", "a")
    for line in range(len(L2v)):
        meanfile.write(str(L2v[line]))
        meanfile.write("\n")
    meanfile.close()
    
    # fig_pathLOSSSVD = '../SVDdata/QoI/'+name+'LossPerformance.eps'
    # plt.plot(LOSS)
    # plt.plot(LOSSval)
    # plt.legend(['Training Temp', 'Validation Temp'], loc='upper right', fontsize=12)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.grid(color="0.8", linewidth=0.5) 
    # plt.xlabel("Epoch", size=14)
    # plt.ylabel("Cost/total loss", size=14)
    # plt.yscale('log')
    # plt.savefig(fig_pathLOSSSVD, bbox_inches='tight', format='eps')
    # plt.close()
    
def UpdateParametersRBF(stanBCname, modes, x_train, x_valid, Tr_train, Tr_valid):
    from scipy.interpolate import RBFInterpolator
    # radial basis function interpolator instance
    rbfi1 = RBFInterpolator(x_train, Tr_train, kernel='inverse_multiquadric', epsilon=3)
    dump(rbfi1, "../SVDdata/"+stanBCname+"/RBF-data.joblib")
    
    yhat1 = rbfi1(torch.from_numpy(np.float32(x_train)))
    v = rbfi1(torch.from_numpy(np.float32(x_train)))
    print("Calculate the L2 norm error from each training and validation sets case")
    L2t = []
    L2v = []
    for i in range(len(Tr_train[:,0])):
        er = np.sqrt(sum((yhat1[i,:] - Tr_train[i,:])**2))
        l2errnorm = er * np.sqrt(0.2*3.0)/np.sqrt(len(yhat1[i,:]))
        L2t.append(l2errnorm)
    for i in range(len(Tr_valid[:,0])):
        er = np.sqrt(sum((v[i,:] - Tr_train[i,:])**2))
        l2errnorm = er * np.sqrt(0.2*3.0)/np.sqrt(len(v[i,:]))
        L2v.append(l2errnorm)
    if os.path.exists('../SVDdata/'+stanBCname+"L2trbf.txt"):
        os.remove('../SVDdata/'+stanBCname+"L2trbf.txt")
    stdfile = open('../SVDdata/'+stanBCname+"L2trbf.txt", "a")
    for line in range(len(L2t)):
        stdfile.write(str(L2t[line]))
        stdfile.write("\n")
    stdfile.close()
    if os.path.exists('../SVDdata/'+stanBCname+"L2vrbf.txt"):
        os.remove('../SVDdata/'+stanBCname+"L2vrbf.txt")
    meanfile = open('../SVDdata/'+stanBCname+"L2vrbf.txt", "a")
    for line in range(len(L2v)):
        meanfile.write(str(L2v[line]))
        meanfile.write("\n")
    meanfile.close()
    
    
def UpdateParametersKriging(stanBCname, modes, x_train, stanBC_train, epochs):
    # kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-15, 1e5))
    min_max_scaler = pp.MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(x_train)
    kernel = ConstantKernel(0.1, (0.01, 10.0)) * Matern(length_scale=1.0, length_scale_bounds=(1e-80, 1e5), nu=0.5)
    #  (
    # DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2
    iter_num = int(epochs/500)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=iter_num, normalize_y=True, copy_X_train=True, random_state=0)
    gpr.fit(x_train_minmax, stanBC_train)
    dump(gpr, "../SVDdata/"+stanBCname+"/GPR-data.joblib")
    
    yhat1, Tr_stdt = gpr.predict(x_train, return_std=True)
    print("Calculate the L2 norm error from each training and validation sets case")
    L2t = []
    for i in range(len(stanBC_train[:,0])):
        er = np.sqrt(sum((yhat1[i,:] - stanBC_train[i,:])**2))
        l2errnorm = er * np.sqrt(0.2*3.0)/np.sqrt(len(yhat1[i,:]))
        L2t.append(l2errnorm)
    if os.path.exists('../SVDdata/'+stanBCname+"L2tkrig.txt"):
        os.remove('../SVDdata/'+stanBCname+"L2tkrig.txt")
    stdfile = open('../SVDdata/'+stanBCname+"L2tkrig.txt", "a")
    for line in range(len(L2t)):
        stdfile.write(str(L2t[line]))
        stdfile.write("\n")
    stdfile.close()
    
def DataQoI(x1, path):
    # Transform the float values of amplitude and wavelength into string to
    # name the folders
    BCList = ['inlet2_U.xy', 'outlet4_U.xy','inlet2_T_p.xy', 'outlet4_T_p.xy']
    # Call for each case data, these files start as lists to append
    inlet = []
    outlet = []
    invelocity = []
    outvelocity = []
    # wallvelocity = []
    # Each case is loaded separately within the loop to append to the total list
    print("Loading data to plot...")
    for p in range(len(x1)):
        for bc in BCList:
            # Built the path to call for each case
            Amp = str(x1[p,0]).replace('.', '-') + '_' + str(x1[p,1]).replace('.', '-')
            WaveL = str(x1[p,2]).replace('.', '-') + '_' + str(x1[p,3]).replace('.', '-')
            filepath = '../caseDatabase/'+"A"+Amp+"_W"+WaveL+'/postProcessing/sample/Helium/1000/'+bc
            # Store the current case, BC & variable data in a numpy array,
            # then transform said array into a tensor for the NN
            Varin = torch.from_numpy(np.loadtxt(filepath))
            # The list.append() requires the data type "float", so the tensor is
            # transformed to float
            Varin = Varin.float()
            # Now append the float tensor into the list of each variable & BC
            if bc == 'inlet2_U.xy':
                invelocity.append(Varin)
            elif bc == 'outlet4_U.xy':
                outvelocity.append(Varin)
            # elif bc == 'wall_U.xy':
            #     wallvelocity.append(Varin)
            elif bc == 'inlet2_T_p.xy':
                inlet.append(Varin)
            elif bc == 'outlet4_T_p.xy':
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
    
    print("Standardizing data for the plots...")
    # BCtensorList = [sui[:,:,1], suo[:,:,1], suw[:,:,1], sin[:,:,1], sout[:,:,1], swall[:,:,1], sin[:,:,2], sout[:,:,2], swall[:,:,2]]
    BCtensorList = [sui[:,:,1], suo[:,:,1], sin[:,:,1], sout[:,:,1], sin[:,:,2], sout[:,:,2]]
    stdList = np.loadtxt('stdData.txt')
    meanList = np.loadtxt('meanData.txt')
    stanBCList = []
    for bct in range(len(BCtensorList)):
        # Built the empty tensor for the standardized data to be built from the variables
        # by taking the standard deviation and mean values from each tensor
        stan = torch.empty(BCtensorList[bct].shape)
        for i in range(0,len(BCtensorList[bct]),1):
            stan[i,:] = (BCtensorList[bct][i,:]-meanList[bct])/stdList[bct]
        stanBCList.append(stan)
    # stanBCList = [stanui, stanuo, stanTi, stanTo, stanpi, stanpo] # boundary  
    stanBCListNames = ["stanui", "stanuo", "stanTi", "stanTo", "stanpi", "stanpo"] # boundary  
    return stanBCList, stanBCListNames, BCtensorList

def DataQoI2(x1, path):
    # Transform the float values of amplitude and wavelength into string to
    # name the folders
    feat = len(x1[0,:])
    x = np.empty([len(x1),feat])
    xin = np.empty([len(x1),feat])
    # Standardize the data for each feature Amp and WaveL
    xmean = np.empty(feat)
    xsd = np.empty(feat)
    # Build the matrix of features, in this case its 2 features with Amp x WaveL snapshots
    for xj in range(feat):
        xmean[xj] = np.mean(x1[:,xj])
        xsd[xj] = np.std(x1[:,xj])
        for xi in range(len(x1)):
            xin[xi,xj] = (x1[xi,xj]-xmean[xj])/xsd[xj]
    x = xin
    BCList = ['inlet2_U.xy', 'outlet4_U.xy','inlet2_T_p.xy', 'outlet4_T_p.xy']
    # Call for each case data, these files start as lists to append
    inlet = []
    outlet = []
    invelocity = []
    outvelocity = []
    # wallvelocity = []
    # Each case is loaded separately within the loop to append to the total list
    print("Loading data to plot...")
    for p in range(len(x1)):
        for bc in BCList:
            # Built the path to call for each case
            Amp = str(x1[p,0]).replace('.', '-') + '_0-0'
            WaveL = str(x1[p,1]).replace('.', '-') + '_0-0'
            filepath = '../caseDatabase/'+"A"+Amp+"_W"+WaveL+'/postProcessing/sample/Helium/1000/'+bc
            # Store the current case, BC & variable data in a numpy array,
            # then transform said array into a tensor for the NN
            Varin = torch.from_numpy(np.loadtxt(filepath))
            # The list.append() requires the data type "float", so the tensor is
            # transformed to float
            Varin = Varin.float()
            # Now append the float tensor into the list of each variable & BC
            if bc == 'inlet2_U.xy':
                invelocity.append(Varin)
            elif bc == 'outlet4_U.xy':
                outvelocity.append(Varin)
            # elif bc == 'wall_U.xy':
            #     wallvelocity.append(Varin)
            elif bc == 'inlet2_T_p.xy':
                inlet.append(Varin)
            elif bc == 'outlet4_T_p.xy':
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
    # BCtensorList = [sui[:,:,1], suo[:,:,1], suw[:,:,1], sin[:,:,1], sout[:,:,1], swall[:,:,1], sin[:,:,2], sout[:,:,2], swall[:,:,2]]
    BCtensorList = [sui[:,:,1], suo[:,:,1], sin[:,:,1], sout[:,:,1], sin[:,:,2], sout[:,:,2]]
    stdList = np.empty([len(BCtensorList)])
    meanList = np.empty([len(BCtensorList)])
    stanBCList = []
    for bct in range(len(BCtensorList)):
        # Built the empty tensor for the standardized data to be built from the variables
        # by taking the standard deviation and mean values from each tensor
        stan = torch.empty(BCtensorList[bct].shape)
        stdList[bct] = np.std(BCtensorList[bct].detach().numpy())
        meanList[bct] = np.mean(BCtensorList[bct].detach().numpy())
        if stdList[bct] == 0.0:
            stdList[bct] = 1.0
            meanList[bct] = 0.0
        # Store the standard deviation and mean values for each BC to calculate
        # later the real variable values
        for i in range(0,len(BCtensorList[bct]),1):
            stan[i,:] = (BCtensorList[bct][i,:]-meanList[bct])/stdList[bct]
        stanBCList.append(stan)
    if os.path.exists("stdData.txt"):
        os.remove("stdData.txt")
    stdfile = open("stdData.txt", "a")
    for line in range(len(stdList)):
        stdfile.write(str(stdList[line]))
        stdfile.write("\n")
    stdfile.close()
    if os.path.exists("meanData.txt"):
        os.remove("meanData.txt")
    meanfile = open("meanData.txt", "a")
    for line in range(len(meanList)):
        meanfile.write(str(meanList[line]))
        meanfile.write("\n")
    meanfile.close()
    # stanBCList = [stanui, stanuo, stanTi, stanTo, stanpi, stanpo] # boundary  
    stanBCListNames = ["stanui", "stanuo", "stanTi", "stanTo", "stanpi", "stanpo"] # boundary  
    return x, stanBCList, stanBCListNames, BCtensorList, xsd, xmean, feat