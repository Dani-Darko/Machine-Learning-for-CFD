#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:13:21 2022

@author: dani
"""

# System, path and runtime information
import os
from importlib import reload
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
import glob

# Call for all the subfunctions needed from the rest of the scripts
import ParametrizationWL
import SimulationWL
import PredictionWL
import OpenFOAMRun
import NeuralNetwork
import QoI
import numpy as np
# Path of this file and subsequently the OpenFOAM and NN data
path = os.getcwd()
path = path.replace('\\', '/')

print ("The current working directory is %s" % path)

# Create the tensors from the old & new database for the NN
x, stanBCList, stanBCListNames, BCtensorList, xsd, xmean, feat = NeuralNetwork.DataDoF(path)
QP, QT = QoI.QoIWL(x, BCtensorList, stanBCListNames, xsd, xmean, feat)

newxin = np.loadtxt("Xnew.txt", delimiter=',')
newx = np.empty([len(newxin[:,0]),len(newxin[0,:])])
for xj in range(len(newxin[0,:])):
    for xi in range(len(newxin[:,0])):
        newx[xi,xj] = (newxin[xi,xj]-xmean[xj])/xsd[xj]
newx = torch.from_numpy(newx)

stanBCList1, stanBCListNames1, BCtensorList1 = NeuralNetwork.DataQoI(newxin, path)
newQP, newQT = QoI.QoIWL(newx, BCtensorList1, stanBCListNames1, xsd, xmean, feat)

xlists =[]
newxin1, newx1 = newxin[:,:], newx[:,:]
newxin1[:,0], newx1[:,0] = newxin[:,0], newx[:,0]
newxin1[:,1], newx1[:,1] = newxin[:,1], newx[:,1]
for i in range(2):
    xlist = list(dict.fromkeys(newxin1[1:,i]))
    xlists.append(xlist)
X, Y = np.meshgrid(xlists[0], xlists[1])

gridQP, gridQT = QoI.gridQoI(xlists, newxin1[:,0:2], newQP, newQT, xsd, xmean)

gridQPp = gridQP
gridQTp = gridQT
QPp = newQP
QTp = newQT

index = [0, 0, 1, 1, 2, 2]
indey = [0, 1, 0, 1, 0, 1]
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
print("Plotting contours for quantities of interest...")
wpars = np.arange(0,1.25,0.25)#
Qi = 0
fig_path = '../SVDdata/cont_A1A2.pdf'
fig_path2 = '../SVDdata/cont_A1A2.png'
fig, ax = plt.subplots(3,2, sharex=True, sharey=True, figsize=(10,10), constrained_layout=True)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
for wpar in wpars:
    cont = (1 - wpar)*gridQPp - wpar*gridQTp
    scat = (1 - wpar)*QPp - wpar*QTp
    surf = ax[index[Qi],indey[Qi]].contourf(X, Y, cont, cmap=cm.jet)
    ax[index[Qi],indey[Qi]].plot(newxin1[1:,0], newxin1[1:,1], 'k.')
    ax[index[Qi],indey[Qi]].axis('on')
    ax[index[Qi],indey[Qi]].set_title(f"$\omega$ = {wpar}", size=14)
    Qi = Qi + 1
ax[-1,0].set_xlabel('Amplitude $A_1$', size=14)
ax[-1,0].xaxis.set_label_coords(1.1, -0.15)
ax[1,0].set_ylabel('Amplitude $A_2$', size=14)
fig.colorbar(surf, ax=ax[-1,:], location='bottom', pad = 0.25)
plt.savefig(fig_path, bbox_inches='tight', format='pdf')
plt.savefig(fig_path2, bbox_inches='tight', format='png')
fig.canvas.draw()
plt.close()