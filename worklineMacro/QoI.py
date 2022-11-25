"""
1. QoIWL.py takes the normalized data from the DoF def to calculate the
    flux in kinetic energy and thermal energy.

@author: Daniela Segura
"""

# # Numpy libraries for array and matrix calculations
import numpy as np

# Matplotlib for plots
import matplotlib.pyplot as plt
from matplotlib import cm

# Constants
xElements = 600
L = 2.0
Cp = 5230

# Create the Parametrization function for worklineMainV1 to call
def QoIWL(x, BCtensorList, stanBCListNames, xsd, xmean, feat):
    Pin = BCtensorList[4]
    Pout = BCtensorList[5]
    Tin= BCtensorList[2]
    Tout = BCtensorList[3]
    Uin = BCtensorList[0]
    Uout = BCtensorList[1]
    QoIp = []
    QoIT = []
    for i in range(0,len(BCtensorList[0])):
        # stanBCList = [stanui, stanuo, stanTi, stanTo, stanpi, stanpo]
        yPin = np.trapz((Pin[i,:] + 1.0/2.0 * Uin[i,:] ** 2) * Uin[i,:], dx=0.00198)
        yPout = np.trapz((Pout[i,:] + 1.0/2.0 * Uout[i,:] ** 2) * Uout[i,:], dx=0.00198)
        yTin = np.trapz((Cp * Tin[i,:]) * Uin[i,:], dx=0.00198)
        yTout = np.trapz((Cp * Tout[i,:]) * Uout[i,:], dx=0.00198)
        QoIp.append(yPin-yPout)
        QoIT.append(yTout-yTin)
    QP = np.asarray(QoIp)
    QT = np.asarray(QoIT)
    
    fig_pathProf = '../SVDdata/QoI/Tprofile.eps'
    plt.plot(Tout[-2,:])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(color="0.8", linewidth=0.5) 
    # plt.xlabel("Epoch", size=14)
    plt.ylabel("Temperature [K]", size=14)
    plt.savefig(fig_pathProf, bbox_inches='tight', format='eps')
    plt.close()
    
    fig_pathProf = '../SVDdata/QoI/Pprofile.eps'
    plt.plot(Pin[-2,:])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(color="0.8", linewidth=0.5) 
    # plt.xlabel("Epoch", size=14)
    plt.ylabel("Pressure [Pa]", size=14)
    plt.savefig(fig_pathProf, bbox_inches='tight', format='eps')
    plt.close()
    
    fig_pathProf = '../SVDdata/QoI/Uprofile.eps'
    plt.plot(Uout[-2,:])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(color="0.8", linewidth=0.5) 
    # plt.xlabel("Epoch", size=14)
    plt.ylabel("Velocity [m/s]", size=14)
    plt.savefig(fig_pathProf, bbox_inches='tight', format='eps')
    plt.close()
    
    return QP, QT

# Create the Parametrization function for worklineMainV1 to call
def calcQoI(xlists, x, xin, BCtensorList, xsd, xmean):
    # Calculate the integral of the Variables Total energy or Thermal energy fluxes
    QoIp = []
    QoIT = []
    for i in range(0,len(BCtensorList[0])):
        # stanBCList = [stanui 0, stanuo 1, stanTi 2, stanTo 3, stanpi 4, stanpo 5]
        yPin = np.trapz((BCtensorList[4][i,:] + 1.0/2.0 * BCtensorList[0][i,:] ** 2) * BCtensorList[0][i,:], dx=0.00198)
        yPout = np.trapz((BCtensorList[5][i,:] + 1.0/2.0 * BCtensorList[1][i,:] ** 2) * BCtensorList[1][i,:], dx=0.00198)
        yTin = np.trapz((Cp * BCtensorList[2][i,:]) * BCtensorList[0][i,:], dx=0.00198)
        yTout = np.trapz((Cp * BCtensorList[3][i,:]) * BCtensorList[1][i,:], dx=0.00198)
        QoIp.append(yPin-yPout)
        QoIT.append(yTout-yTin)
    QP = np.asarray(QoIp)
    QT = np.asarray(QoIT)
    X, Y = np.meshgrid(xlists[0], xlists[1])
    gridQP = np.zeros((len(xlists[0]),len(xlists[1])))
    gridQT = np.zeros((len(xlists[0]),len(xlists[1])))
    for i in range(len(xlists[0])):
        for j in range(len(xlists[1])):
            indexX, = np.where((xlists[0][i]==xin[:,0]))
            indexY, = np.where((xlists[1][j]==xin[:,1]))
            gridQP[i,j] = QP[np.intersect1d(indexX, indexY)]
            gridQT[i,j] = QT[np.intersect1d(indexX, indexY)]
    # index0 = np.where((x[:,0]==0.0))
    index0 = 0
    QP0 = QP[index0]
    QT0 = QT[index0]
    stanQP = (gridQP)/QP0
    stanQT = (gridQT)/QT0
    sQP = QP/QP0
    sQT = QT/QT0
    return stanQP, stanQT, sQP, sQT

def gridQoI(xlists, x, QP, QT, xsd, xmean):
    X, Y = np.meshgrid(xlists[0], xlists[1])
    gridQP = np.zeros((len(xlists[0]),len(xlists[1])))
    gridQT = np.zeros((len(xlists[0]),len(xlists[1])))
    for i in range(len(xlists[0])):
        for j in range(len(xlists[1])):
            indexX, = np.where((xlists[0][i]==x[:,0]))
            indexY, = np.where((xlists[1][j]==x[:,1]))
            gridQP[i,j] = QP[np.intersect1d(indexX, indexY)]
            gridQT[i,j] = QT[np.intersect1d(indexX, indexY)]
    # index0 = np.where((x[:,0]==0.0))
    index0 = 0
    QP0 = QP[index0]
    QT0 = QT[index0]
    
    stanQP = (gridQP)/QP0
    stanQT = (gridQT)/QT0
    
    return stanQP, stanQT