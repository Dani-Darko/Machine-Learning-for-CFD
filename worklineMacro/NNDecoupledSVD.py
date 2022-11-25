#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:43:34 2021

@author: dani
"""

from scipy.optimize import curve_fit
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import math as mt
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.stats import kde
import matplotlib as mpl
import time
import random
from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd
import sys
 
#Experimental x and y data points
path = os.getcwd()
path = path.replace('\\', '/')

print ("The current working directory is %s" % path)

dat = 201


plt.rcParams["font.family"] = "serif"

x0 = np.arange(0.0005,0.0025,0.0005) # Amplitude
x1 = np.arange(4,21,4) # Wavelength
x = np.empty([len(x0)*len(x1)+1,2])
xin = np.empty([len(x0)*len(x1),2])
xsd0 = np.std(x0)
xsd1 = np.std(x1)
xmean0 = np.mean(x0)
xmean1 = np.mean(x1)
shapes = -1
x[0,0] = 0.0
x[0,1] = 0.0
for xi in range(0,len(x0),1):
    for xj in range(0,len(x1),1):
        shapes = shapes + 1
        xin[shapes,0] = (x0[xi]-xmean0)/xsd0
        xin[shapes,1] = (x1[xj]-xmean1)/xsd1
x[1:,0] = xin[:,0]
x[1:,1] = xin[:,1]

# define the name of the directory to be created
patharray = ['A0005', 'A001', 'A0015', 'A002']
path2 = ['W4', 'W8', 'W12', 'W16', 'W20']
inlet = []
outlet = []
symmetry = []
wall = []
invelocity = []
outvelocity = []
symvelocity = []
wallvelocity = []

# Array of values is [T, U:x, U:y, U:z, p, p_prgh, rho,vtkValidPointMask, arc_length, points:x, points:y, points:z]
filepath = path+'/A000/postProcessing/sample/Helium/1000/'
inletufile = filepath+'inlet_U.xy'
outletufile = filepath+'outlet_U.xy'
symmetryufile = filepath+'symmetry_U.xy'
wallufile = filepath+'wall_U.xy'
inletfile = filepath+'inlet_T_p.xy'
outletfile = filepath+'outlet_T_p.xy'
symmetryfile = filepath+'symmetry_T_p.xy'
wallfile = filepath+'wall_T_p.xy'
inuin = torch.from_numpy(np.loadtxt(fname = inletufile))
outuin = torch.from_numpy(np.loadtxt(fname = outletufile))
symuin = torch.from_numpy(np.loadtxt(fname = symmetryufile))
walluin = torch.from_numpy(np.loadtxt(fname = wallufile))
inin = torch.from_numpy(np.loadtxt(fname = inletfile))
outin = torch.from_numpy(np.loadtxt(fname = outletfile))
symin = torch.from_numpy(np.loadtxt(fname = symmetryfile))
wallin = torch.from_numpy(np.loadtxt(fname = wallfile))
inuin = inuin.float()
outuin = outuin.float()
symuin = symuin.float()
walluin = walluin.float()
inin = inin.float()
outin = outin.float()
symin = symin.float()
wallin = wallin.float()
invelocity.append(inuin)
outvelocity.append(outuin)
symvelocity.append(symuin)
wallvelocity.append(walluin)
inlet.append(inin)
outlet.append(outin)    
symmetry.append(symin)
wall.append(wallin)

for p in patharray:
    for p2 in path2:
        filepath = path+'/'+p+'/'+p2+'/postProcessing/sample/Helium/1000/'
        inletufile = filepath+'inlet_U.xy'
        outletufile = filepath+'outlet_U.xy'
        symmetryufile = filepath+'symmetry_U.xy'
        wallufile = filepath+'wall_U.xy'
        inletfile = filepath+'inlet_T_p.xy'
        outletfile = filepath+'outlet_T_p.xy'
        symmetryfile = filepath+'symmetry_T_p.xy'
        wallfile = filepath+'wall_T_p.xy'
        inuin = torch.from_numpy(np.loadtxt(fname = inletufile))
        outuin = torch.from_numpy(np.loadtxt(fname = outletufile))
        symuin = torch.from_numpy(np.loadtxt(fname = symmetryufile))
        walluin = torch.from_numpy(np.loadtxt(fname = wallufile))
        inin = torch.from_numpy(np.loadtxt(fname = inletfile))
        outin = torch.from_numpy(np.loadtxt(fname = outletfile))
        symin = torch.from_numpy(np.loadtxt(fname = symmetryfile))
        wallin = torch.from_numpy(np.loadtxt(fname = wallfile))
        inuin = inuin.float()
        outuin = outuin.float()
        symuin = symuin.float()
        walluin = walluin.float()
        inin = inin.float()
        outin = outin.float()
        symin = symin.float()
        wallin = wallin.float()
        invelocity.append(inuin)
        outvelocity.append(outuin)
        symvelocity.append(symuin)
        wallvelocity.append(walluin)
        inlet.append(inin)
        outlet.append(outin)    
        symmetry.append(symin)
        wall.append(wallin)
sui = torch.stack(invelocity)
suo = torch.stack(outvelocity)
sus = torch.stack(symvelocity)
suw = torch.stack(wallvelocity)
sin = torch.stack(inlet)
sout = torch.stack(outlet)
ssym = torch.stack(symmetry)
swall = torch.stack(wall)

pi = sin[:,:,2]
po = sout[:,:,2]
ps = ssym[:,:,2]
pw = swall[:,:,2]
Ti = sin[:,:,1]
To = sout[:,:,1]
Ts = ssym[:,:,1]
Tw = swall[:,:,1]
ui = sui[:,:,1]
uo = suo[:,:,1]
us = sus[:,:,1]
uw = suw[:,:,1]
pin = torch.cat((pi, po, pw), 1)
Tin = torch.cat((Ti, To, Tw), 1)

stanTi = torch.empty(Ti.shape)
stanTo = torch.empty(To.shape)
stanTs = torch.empty(Ts.shape)
stanTw = torch.empty(Tw.shape)
stanpi = torch.empty(pi.shape)
stanpo = torch.empty(po.shape)
stanps = torch.empty(ps.shape)
stanpw = torch.empty(pw.shape)
stanui = torch.empty(ui.shape)
stanuo = torch.empty(uo.shape)
stanus = torch.empty(us.shape)
stanuw = torch.empty(uw.shape)
        
psdi = np.std(pi.detach().numpy())
pmeani = np.mean(pi.detach().numpy())
psdo = np.std(po.detach().numpy())
pmeano = np.mean(po.detach().numpy())
psds = np.std(ps.detach().numpy())
pmeans = np.mean(ps.detach().numpy())
psdw = np.std(pw.detach().numpy())
pmeanw = np.mean(pw.detach().numpy())
Tsdi = np.std(Ti.detach().numpy())
Tmeani = np.mean(Ti.detach().numpy())
Tsdo = np.std(To.detach().numpy())
Tmeano = np.mean(To.detach().numpy())
Tsds = np.std(Ts.detach().numpy())
Tmeans = np.mean(Ts.detach().numpy())
Tsdw = np.std(Tw.detach().numpy())
Tmeanw = np.mean(Tw.detach().numpy())
usdi = np.std(ui.detach().numpy())
umeani = np.mean(ui.detach().numpy())
usdo = np.std(uo.detach().numpy())
umeano = np.mean(uo.detach().numpy())
usds = np.std(us.detach().numpy())
umeans = np.mean(us.detach().numpy())
usdw = np.std(uw.detach().numpy())
umeanw = np.mean(uw.detach().numpy())

for i in range(0,len(x),1):
    stanTi[i,:] = (Ti[i,:]-Tmeani)/Tsdi
    stanTo[i,:] = (To[i,:]-Tmeano)/Tsdo
    stanTs[i,:] = (Ts[i,:]-Tmeans)
    stanTw[i,:] = (Tw[i,:]-Tmeanw)/Tsdw
    stanpi[i,:] = (pi[i,:]-pmeani)/psdi
    stanpo[i,:] = (po[i,:]-pmeano)/psdo
    stanps[i,:] = (ps[i,:]-pmeans)/psds
    stanpw[i,:] = (pw[i,:]-pmeanw)/psdw
    stanui[i,:] = (ui[i,:]-umeani)/usdi
    stanuo[i,:] = (uo[i,:]-umeano)/usdo
    stanus[i,:] = (us[i,:]-umeans)/usds
    stanuw[i,:] = (uw[i,:]-umeanw)/usdw


Tw_cases = [stanTi, stanTo, stanTs, stanTw, stanpi, stanpo, stanps, stanpw, stanui, stanuo, stanus, stanuw] # boundary  
u_cases = [1] # output variable
outDat = dat
names = ['Tin', 'Tout', 'Tsym', 'Twall', 'Pin', 'Pout', 'Psym', 'Pwall', 'Uin', 'Uout', 'Usym', 'Uwall']
inames = 0

# define the name of the directory to be created
for iTw in range(0,len(Tw_cases)):
    for iu in u_cases:
        x0t = np.arange(0.0005,0.0025,0.0005) # Amplitude
        x1t = np.arange(4,21,4) # Wavelength
        xtest = np.empty([len(x0t)*len(x1t)+1,2])
        xt = np.empty([len(x0t)*len(x1t),2])
        xsd0t = np.std(x0t)
        xsd1t = np.std(x1t)
        xmean0t = np.mean(x0t)
        xmean1t = np.mean(x1t)
        shapes = -1
        xtest[0,0] = 0.0
        xtest[0,1] = 0.0
        for xti in range(0,len(x0t),1):
            for xtj in range(0,len(x1t),1):
                shapes = shapes + 1
                xt[shapes,0] = (x0t[xti]-xmean0t)/xsd0t
                xt[shapes,1] = (x1t[xtj]-xmean1t)/xsd1t
        xtest[1:,0] = xt[:,0]
        xtest[1:,1] = xt[:,1]
        
        
        Ttest = torch.from_numpy(np.float32(Tw_cases[iTw]))
        
        shapes = -1
        Twshape = np.empty([len(xtest)])
        for xci in range(0,len(x0t),1):
            for xcj in range(0,len(x1t),1):
                shapes = shapes + 1
                for mc in range(0,dat,1):
                    Twshape[shapes] = x1t[xcj]
        shapes = np.arange(0,dat,1)
        # fig_pathM = '2D-ontour-'+names[inames]+'.eps'
        # fig, ax = plt.subplots()
        # im = ax.contourf(shapes, Twshape, Ttest, cmap='jet')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.ylabel("Outer wall T", size=14)
        # plt.xlabel("DoF", size=14)
        # plt.colorbar(im)
        # plt.savefig(fig_pathM, bbox_inches='tight', format='eps')
        # plt.close()
        
        # Xs, YTw = np.meshgrid(shapes, Twshape)
        # fig_pathM = '3D-contour-'+names[inames]+'.eps'
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # surf = ax.plot_surface(Xs, YTw, Ttest.detach().numpy(), cmap='jet', vmin=-0.5, vmax=0.5)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.ylabel("Outer wall T", size=14)
        # plt.xlabel("DoF", size=14)
        # fig.colorbar(surf)
        # plt.savefig(fig_pathM, bbox_inches='tight', format='eps')
        # plt.close()
        
         # define a matrix
        A = Ttest.detach().numpy()
        # Singular-value decomposition
        U, s, VT = svd(A)
        print(U.shape, s.shape, VT.shape)
        m = np.arange(0.0, 5.0, 1.0)
        modes = 5
        fig_pathModes = 'Modes-'+names[inames]+'.eps'
        plt.scatter(m, s[:len(m)])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(color="0.8", linewidth=0.5) 
        # plt.title('First 50 modes with ' + str(len(x1t)*len(x0t))+' snapshots')
        plt.ylabel("Relative singular value", size=14)
        plt.xlabel("Mode", size=14)
        plt.yscale('log')
        plt.savefig(fig_pathModes, bbox_inches='tight', format='eps')
        plt.close()
        # create m x n Sigma matrix
        Sigma = zeros((A.shape[0], A.shape[1]))
        # populate Sigma with n x n diagonal matrix
        if len(x0t)*len(x1t) > 200:
            Sigma[:A.shape[1], :A.shape[1]] = diag(s)
        else:
            Sigma[:A.shape[0], :A.shape[0]] = diag(s)
        # select
        n_elements = modes
        Sigma = Sigma[:, :n_elements]
        VT = VT[:n_elements, :]
        # transform
        Trin = U.dot(Sigma)
        name = 'VT-SVD-'+names[inames]+'.csv'
        np.savetxt(name, VT, delimiter=",")
        Tr = np.empty(Trin.shape)
        Trsd = np.std(Trin)
        Trmean = np.mean(Trin)
        for tri in range(0,len(Trin),1):
            for mj in range(0,modes,1):
                Tr[tri,mj] = (Trin[tri,mj]-Trmean)/Trsd
        print(U.shape)
        sys.exit()
        
        dataset = Ttest
        batch_size = len(xt[:,1]) * len(xt[:,0]) -1
        validation_split = 0.35
                
        # Creating data indices for training and validation splits:
        bT = np.asarray(np.where(xin[:,1] == min(xin[:,1])))
        tT = np.asarray(np.where(xin[:,1] == max(xin[:,1])))
        bT = list(np.squeeze(bT, axis=(0,)))
        tT = list(np.squeeze(tT, axis=(0,)))

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        indices2 = indices
        for b in bT:
            indices2.remove(b)
        for t in tT:
            indices2.remove(t)

        split = int(np.floor(validation_split * dataset_size))
        random.shuffle(indices2)
        train_indices, val_indices = indices2[split:], indices2[:split]
        for b in bT:
            train_indices.append(b)
        for t in tT:
            train_indices.append(t)

        x_train = torch.from_numpy(np.float32(xtest[train_indices]))
        x_valid = torch.from_numpy(np.float32(xtest[val_indices]))
        T_train = Ttest[train_indices,:]
        T_valid = Ttest[val_indices,:]
        Tr_train = torch.from_numpy(np.float32(Tr[train_indices,:]))
        Tr_valid = torch.from_numpy(np.float32(Tr[val_indices,:]))
    
        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        hidden_sizes = 32
        D_in = 2
        D_out = modes

        # Create a customized linear
                
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Inputs to hidden layer linear transformation
                self.hidden1 = torch.nn.Linear(D_in, hidden_sizes)
                # Hidden layer 
                self.hidden2 = torch.nn.Linear(hidden_sizes, hidden_sizes)
                # hidden layer to hidden output
                self.output = torch.nn.Linear(hidden_sizes, D_out)
                self.hidden1.weight.data = torch.rand(hidden_sizes, D_in)*0.5
                self.hidden2.weight.data = torch.rand(hidden_sizes, hidden_sizes)*0.5

                # Define sigmoid activation
                self.sigmoid = torch.nn.Sigmoid()
                self.GELU = torch.nn.Tanh()

            def forward(self, x):
                # Pass the input tensor through each of our operations
                a1 = self.hidden1(x)
                a1 = self.sigmoid(a1)
                a1 = self.hidden2(a1)
                a1 = self.sigmoid(a1)
                # a1 = self.hidden2(a1)
                # a1 = self.sigmoid(a1)
                # a1 = self.hidden2(a1)
                # a1 = self.sigmoid(a1)
                # a1 = self.hidden2(a1)
                # a1 = self.sigmoid(a1)
                # a1 = self.hidden2(a1)
                # a1 = self.sigmoid(a1)
                # a1 = self.hidden2(a1)
                # a1 = self.sigmoid(a1)
                # a1 = self.hidden2(a1)
                # a1 = self.sigmoid(a1)
                # a1 = self.hidden2(a1)
                # a1 = self.sigmoid(a1)
                g = self.output(a1)

                return g
        network1 = Network()
        
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
        optimizer = optim.Rprop(network1.parameters())
        
        loss_fn = torch.nn.MSELoss()
                    
        # Train the model
        epochs = 2000
        stop_criteria = 0.000000001
        LOSS = []
        LOSSval = []
        # Checkpoint
        checkpoint_path='SVD-T-outData-NN-'+names[inames]+'.pt'
        checkpoint={'epoch':None,'model_state_dict':None ,'optimizer_state_dict':None ,'loss': None, 'lossv': None}
        # Decoupled NNs, 2 NN
        for epoch in range(epochs):
            # yhat1 = network1(x_train)
            yhat1 = network1(x_train)
            # loss1 = loss_fn(yhat1, p_train[:,i].unsqueeze(1))
            loss1 = loss_fn(yhat1, Tr_train)
            LOSS.append(loss1.item())
            # LOSSp.append(loss1.item())
            # pv = network1(x_valid)
            v = network1(x_valid)
            # loss1v = loss_fn(pv, p_valid[:,i].unsqueeze(1))
            loss2v = loss_fn(v, Tr_valid)
            LOSSval.append(loss2v.item())
            # LOSSvalp.append(loss1v.item())
            optimizer.zero_grad()
            # loss1.backward()
            loss1.backward()
            optimizer.step()
            # if LOSST[-1] < stop_criteria and LOSSp[-1] < stop_criteria :
            if LOSS[-1] < stop_criteria:
                print('Minimal loss criteria has been reached', epoch)
                break
        checkpoint['epoch']=epochs
        checkpoint['model_state_dict']=network1.state_dict()
        checkpoint['optimizer_state_dict']= optimizer.state_dict()
        checkpoint['loss']=LOSS
        checkpoint['lossv']=LOSSval
        torch.save(checkpoint, checkpoint_path)
        
        fig_pathLOSSSVD = 'LOSSvsEpoch-'+names[inames]+'.eps'
        plt.plot(LOSS)
        # plt.plot(LOSSp)
        plt.plot(LOSSval)
        # plt.plot(LOSSvalp)
        plt.legend(['Training Temp', 'Validation Temp'], loc='upper right', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(color="0.8", linewidth=0.5) 
        # plt.title('Loss for ' + str(len(x1t))+' temperature iterations, '+str(len(x0t))+' velocity iterations')
        plt.xlabel("Epoch", size=14)
        plt.ylabel("Cost/total loss", size=14)
        plt.yscale('log')
        plt.savefig(fig_pathLOSSSVD, bbox_inches='tight', format='eps')
        plt.close()
        # Total loss for decoupled NN
        print('Total singular value loss is ', LOSS[-1])
        print('Total singular value vs validation set is ', LOSSval[-1])
        yat = network1(torch.from_numpy(np.float32(xtest))).detach().numpy()
        yhat = np.empty(yat.shape)
        for tri in range(0,len(yat),1):
            for mj in range(0,modes,1):
                yhat[tri,mj] = yat[tri,mj]*Trsd+Trmean
        Apreds = yhat.dot(VT)
        PredsSigma = np.linalg.lstsq(U, yhat)
        PredsTr = PredsSigma[0]
        
        j00 = np.where(xtest[:,0] == min(xtest[:,0]))
        j01 = np.where(xtest[:,0] == max(xtest[:,0]))
        j10 = np.where(xtest[:,1] == min(xtest[:,1]))
        j11 = np.where(xtest[:,1] == max(xtest[:,1]))
        
        jtot = np.empty(4)
        jtot[0] = np.intersect1d(j00,j10)
        jtot[1] = np.intersect1d(j01,j10)
        jtot[2] = np.intersect1d(j00,j11)
        jtot[3] = np.intersect1d(j01,j11)
        
        jint = 0
        for j in jtot:
            jint = jint + 1
            j = int(j)
            fig_pathT = str(jint) + '-'+names[inames]+'.eps'
            plt.plot(Ttest[j,:])
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(color="0.8", linewidth=0.5) 
            plt.xlabel("Degrees of Freedom", size=14)
            plt.ylabel("Normalized variable", size=14)
            plt.savefig(fig_pathT, bbox_inches='tight', format='eps')
            plt.close()
            # fig_pathTvsPredsT = 'Temperature-DoF-' + str(jint) + '-NN-'+str(len(x1t))+'-Tw-'+str(len(x0t))+'-u.eps'
            # TplotDoF = Ttest[j,:] - T[j,:]
            # # plt.plot(T[j,:].detach().numpy())
            # plt.plot(abs(TplotDoF))
            # plt.legend(['Closed form interpolant', 'Neural network prediction'], loc='upper right', fontsize=12)
            # plt.xticks(fontsize=10)
            # plt.yticks(fontsize=10)
            # # plt.title('Temperature degrees of freedom with  ' + str(len(x1t)*len(x0t))+' snapshots')
            # plt.xlabel("Degrees of Freedom", size=14)
            # plt.ylabel("Difference vs benchmark", size=14)
            # plt.yscale('log')
            # plt.savefig(fig_pathTvsPredsT, bbox_inches='tight', format='eps')
            # plt.close()
            fig_pathTvsPredsSVDT = 'Difference' + str(jint) + names[inames] + '.eps'
            TplotSVD = Ttest[j,:] - Apreds[j,:]
            # plt.plot(Apreds[j,:])
            plt.plot(abs(TplotSVD))
            # plt.legend(['Neural network prediction difference vs numerical'], loc='upper right', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(color="0.8", linewidth=0.5) 
            plt.xlabel("DoF", size=14)
            plt.ylabel("Difference vs benchmark", size=14)
            plt.yscale('log')
            plt.savefig(fig_pathTvsPredsSVDT, bbox_inches='tight', format='eps')
            plt.close()
        inames = inames + 1
