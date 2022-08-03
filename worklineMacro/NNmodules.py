#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the rest of the NeuralNetwork.py file, just to run the other with no problems.

@author: dani
"""

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
 
# Neural Network module for the SVD
def NNSVD(hidden_sizes, D_in, D_out):
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
            g = self.output(a1)
            
            return g
        
    network = Network()
    return network

# Neural Network module for the DoF
def NNDoF(hidden_sizes, D_in, D_out):
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
            g = self.output(a1)
            
            return g
        
    network = Network()
    return network

        
        
        # fig_pathLOSSSVD = 'LOSSvsEpoch-'+names[inames]+'.eps'
        # plt.plot(LOSS)
        # # plt.plot(LOSSp)
        # plt.plot(LOSSval)
        # # plt.plot(LOSSvalp)
        # plt.legend(['Training Temp', 'Validation Temp'], loc='upper right', fontsize=12)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.grid(color="0.8", linewidth=0.5) 
        # # plt.title('Loss for ' + str(len(x1t))+' temperature iterations, '+str(len(x0t))+' velocity iterations')
        # plt.xlabel("Epoch", size=14)
        # plt.ylabel("Cost/total loss", size=14)
        # plt.yscale('log')
        # plt.savefig(fig_pathLOSSSVD, bbox_inches='tight', format='eps')
        # plt.close()
        # # Total loss for decoupled NN
        # print('Total singular value loss is ', LOSS[-1])
        # print('Total singular value vs validation set is ', LOSSval[-1])
        # yat = network1(torch.from_numpy(np.float32(xtest))).detach().numpy()
        # yhat = np.empty(yat.shape)
        # for tri in range(0,len(yat),1):
        #     for mj in range(0,modes,1):
        #         yhat[tri,mj] = yat[tri,mj]*Trsd+Trmean
        # Apreds = yhat.dot(VT)
        # PredsSigma = np.linalg.lstsq(U, yhat)
        # PredsTr = PredsSigma[0]
        
        # j00 = np.where(xtest[:,0] == min(xtest[:,0]))
        # j01 = np.where(xtest[:,0] == max(xtest[:,0]))
        # j10 = np.where(xtest[:,1] == min(xtest[:,1]))
        # j11 = np.where(xtest[:,1] == max(xtest[:,1]))
        
        # jtot = np.empty(4)
        # jtot[0] = np.intersect1d(j00,j10)
        # jtot[1] = np.intersect1d(j01,j10)
        # jtot[2] = np.intersect1d(j00,j11)
        # jtot[3] = np.intersect1d(j01,j11)
        
        # jint = 0
        # for j in jtot:
        #     jint = jint + 1
        #     j = int(j)
        #     fig_pathT = str(jint) + '-'+names[inames]+'.eps'
        #     plt.plot(Ttest[j,:])
        #     plt.xticks(fontsize=10)
        #     plt.yticks(fontsize=10)
        #     plt.grid(color="0.8", linewidth=0.5) 
        #     plt.xlabel("Degrees of Freedom", size=14)
        #     plt.ylabel("Normalized variable", size=14)
        #     plt.savefig(fig_pathT, bbox_inches='tight', format='eps')
        #     plt.close()
        #     # fig_pathTvsPredsT = 'Temperature-DoF-' + str(jint) + '-NN-'+str(len(x1t))+'-Tw-'+str(len(x0t))+'-u.eps'
        #     # TplotDoF = Ttest[j,:] - T[j,:]
        #     # # plt.plot(T[j,:].detach().numpy())
        #     # plt.plot(abs(TplotDoF))
        #     # plt.legend(['Closed form interpolant', 'Neural network prediction'], loc='upper right', fontsize=12)
        #     # plt.xticks(fontsize=10)
        #     # plt.yticks(fontsize=10)
        #     # # plt.title('Temperature degrees of freedom with  ' + str(len(x1t)*len(x0t))+' snapshots')
        #     # plt.xlabel("Degrees of Freedom", size=14)
        #     # plt.ylabel("Difference vs benchmark", size=14)
        #     # plt.yscale('log')
        #     # plt.savefig(fig_pathTvsPredsT, bbox_inches='tight', format='eps')
        #     # plt.close()
        #     fig_pathTvsPredsSVDT = 'Difference' + str(jint) + names[inames] + '.eps'
        #     TplotSVD = Ttest[j,:] - Apreds[j,:]
        #     # plt.plot(Apreds[j,:])
        #     plt.plot(abs(TplotSVD))
        #     # plt.legend(['Neural network prediction difference vs numerical'], loc='upper right', fontsize=12)
        #     plt.xticks(fontsize=10)
        #     plt.yticks(fontsize=10)
        #     plt.grid(color="0.8", linewidth=0.5) 
        #     plt.xlabel("DoF", size=14)
        #     plt.ylabel("Difference vs benchmark", size=14)
        #     plt.yscale('log')
        #     plt.savefig(fig_pathTvsPredsSVDT, bbox_inches='tight', format='eps')
        #     plt.close()
        # inames = inames + 1