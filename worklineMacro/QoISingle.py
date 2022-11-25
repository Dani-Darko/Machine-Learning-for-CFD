"""
1. ParametrizationWL.py takes geometry parameter values from 
    the user to input into functions and built the points coordinates
    for the OpenFOAM files

@author: Daniela Segura
"""
# Neural networks libraries
import torch
from torch import nn,optim

# # Numpy libraries for array and matrix calculations
import numpy as np
from numpy import array
from numpy import diag
from numpy import zeros

# Matplotlib for plots
import matplotlib.pyplot as plt

import glob
from scipy.linalg import svd
from scipy.integrate import simps


# Constants
Cp = 5230


# Create a list with both old & new data sampling
DirList = ["A0-0_W0", "A0-001_W4", "A0-001_W8", "A0-001_W16", "A0-001_W32", "A0-002_W4", "A0-002_W8", "A0-002_W16", "A0-002_W32", "A0-003_W4", "A0-003_W8", "A0-003_W16", "A0-003_W32", "A0-004_W4", "A0-004_W8", "A0-004_W16", "A0-004_W32"]
BCList = ['inlet_T_p.xy', 'outlet_T_p.xy', 'inlet_U.xy', 'outlet_U.xy', 'inlet2_T_p.xy', 'outlet2_T_p.xy', 'inlet2_U.xy', 'outlet2_U.xy','inlet3_T_p.xy', 'outlet3_T_p.xy', 'inlet3_U.xy', 'outlet3_U.xy','inlet4_T_p.xy', 'outlet4_T_p.xy', 'inlet4_U.xy', 'outlet4_U.xy']
# Call for the amplitude and wavelength data from each and every case in existance
Amp = [0.0, 0.001, 0.002]
WaveL = [0, 4, 8]
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
invelocity = []
outvelocity = []
inlet2 = []
outlet2 = []
invelocity2 = []
outvelocity2 = []
inlet3 = []
outlet3 = []
invelocity3 = []
outvelocity3 = []
inlet4 = []
outlet4 = []
invelocity4 = []
outvelocity4 = []
# Each case is loaded separately within the loop to append to the total list
for p in DirList:
    for bc in BCList:
        # Built the path to call for each case
        filepath = '../HPCtest/caseDatabase/'+p+'/postProcessing/sample/Helium/5000/'+bc
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
        elif bc == 'inlet_T_p.xy':
            inlet.append(Varin)
        elif bc == 'outlet_T_p.xy':
            outlet.append(Varin)
        elif bc == 'inlet2_U.xy':
            invelocity2.append(Varin)
        elif bc == 'outlet2_U.xy':
            outvelocity2.append(Varin)
        elif bc == 'inlet2_T_p.xy':
            inlet2.append(Varin)
        elif bc == 'outlet2_T_p.xy':
            outlet2.append(Varin)
        elif bc == 'inlet3_U.xy':
            invelocity3.append(Varin)
        elif bc == 'outlet3_U.xy':
            outvelocity3.append(Varin)
        elif bc == 'inlet3_T_p.xy':
            inlet3.append(Varin)
        elif bc == 'outlet3_T_p.xy':
            outlet3.append(Varin)
        elif bc == 'inlet4_U.xy':
            invelocity4.append(Varin)
        elif bc == 'outlet4_U.xy':
            outvelocity4.append(Varin)
        elif bc == 'inlet4_T_p.xy':
            inlet4.append(Varin)
        elif bc == 'outlet4_T_p.xy':
            outlet4.append(Varin)
# Transform the lists into a matrix of tensors of shape U = snapshots x DoF x coordinate
# while TP = snapshots x DoF x coor,T,P location in file
sui = torch.stack(invelocity)
suo = torch.stack(outvelocity)
sin = torch.stack(inlet)
sout = torch.stack(outlet)
sui2 = torch.stack(invelocity2)
suo2 = torch.stack(outvelocity2)
sin2 = torch.stack(inlet2)
sout2 = torch.stack(outlet2)
sui3 = torch.stack(invelocity3)
suo3 = torch.stack(outvelocity3)
sin3 = torch.stack(inlet3)
sout3 = torch.stack(outlet3)
sui4 = torch.stack(invelocity4)
suo4 = torch.stack(outvelocity4)
sin4 = torch.stack(inlet4)
sout4 = torch.stack(outlet4)
# fig_pathM = 'Inletp.eps'
# for i in range(0,len(DirList)):
#     plt.plot(sin[i,:,2], "--")
#     plt.plot(sin2[i,:,2], "--")
#     plt.plot(sin3[i,:,2], "--")
#     plt.plot(sin4[i,:,2], "--")
#     # plt.yscale("log")
#     # plt.xscale("log")
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(color="0.8", linewidth=0.5) 
# plt.ylabel("Pressure [Pa]", size=14)
# plt.xlabel("DoF", size=14)
# plt.legend(['Smooth surface x=0.1', 'Smooth surface x=0.2', 'Smooth surface x=0.3', 'Smooth surface x=0.4',
#             'A=0.001,W=4  x=0.1', 'A=0.001,W=4  x=0.2', 'A=0.001,W=4  x=0.3', 'A=0.001,W=4  x=0.4', 
#             'A=0.001,W=16  x=0.1', 'A=0.001,W=16  x=0.2', 'A=0.001,W=16  x=0.3', 'A=0.001,W=16  x=0.4', 
#             'A=0.002,W=4  x=0.1', 'A=0.002,W=4  x=0.2', 'A=0.002,W=4  x=0.3', 'A=0.002,W=4  x=0.4', 
#             'A=0.002,W=16  x=0.1', 'A=0.002,W=16  x=0.2', 'A=0.002,W=16  x=0.3', 'A=0.002,W=16  x=0.4'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
# plt.savefig(fig_pathM, bbox_inches='tight', format='eps')
# plt.close()
# fig_pathM = 'Outletp.eps'
# for i in range(0,len(DirList)):
#     plt.plot(sout[i,:,2], "--")
#     plt.plot(sout2[i,:,2], "--")
#     plt.plot(sout3[i,:,2], "--")
#     plt.plot(sout4[i,:,2], "--")
# # plt.yscale("log")
# # plt.xscale("log")
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(color="0.8", linewidth=0.5) 
# plt.ylabel("Pressure [Pa]", size=14)
# plt.xlabel("DoF", size=14)
# plt.legend(['Smooth surface x=0.1', 'Smooth surface x=0.2', 'Smooth surface x=0.3', 'Smooth surface x=0.4',
#             'A=0.001,W=4  x=0.1', 'A=0.001,W=4  x=0.2', 'A=0.001,W=4  x=0.3', 'A=0.001,W=4  x=0.4', 
#             'A=0.001,W=16  x=0.1', 'A=0.001,W=16  x=0.2', 'A=0.001,W=16  x=0.3', 'A=0.001,W=16  x=0.4', 
#             'A=0.002,W=4  x=0.1', 'A=0.002,W=4  x=0.2', 'A=0.002,W=4  x=0.3', 'A=0.002,W=4  x=0.4', 
#             'A=0.002,W=16  x=0.1', 'A=0.002,W=16  x=0.2', 'A=0.002,W=16  x=0.3', 'A=0.002,W=16  x=0.4'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
# plt.savefig(fig_pathM, bbox_inches='tight', format='eps')
# plt.close()
# fig_pathM = 'InletT.eps'
# for i in range(12,13):
# # for i in range(0,len(DirList)):
#     print(i)
#     plt.plot(sin[i,:,1], "--")
#     plt.plot(sin2[i,:,1], "--")
#     plt.plot(sin3[i,:,1], "--")
#     plt.plot(sin4[i,:,1], "--")
#     # plt.yscale("log")
#     # plt.xscale("log")
# # plt.xticks(fontsize=10)
# # plt.yticks(fontsize=10)
# # plt.grid(color="0.8", linewidth=0.5) 
# # plt.ylabel("Temperature [K]", size=14)
# # plt.xlabel("DoF", size=14)
# # plt.legend(['Smooth surface x=0.1', 'Smooth surface x=0.2', 'Smooth surface x=0.3', 'Smooth surface x=0.4',
# #             'A=0.001,W=4  x=0.1', 'A=0.001,W=4  x=0.2', 'A=0.001,W=4  x=0.3', 'A=0.001,W=4  x=0.4', 
# #             'A=0.001,W=16  x=0.1', 'A=0.001,W=16  x=0.2', 'A=0.001,W=16  x=0.3', 'A=0.001,W=16  x=0.4', 
# #             'A=0.002,W=4  x=0.1', 'A=0.002,W=4  x=0.2', 'A=0.002,W=4  x=0.3', 'A=0.002,W=4  x=0.4', 
# #             'A=0.002,W=16  x=0.1', 'A=0.002,W=16  x=0.2', 'A=0.002,W=16  x=0.3', 'A=0.002,W=16  x=0.4'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
# # plt.savefig(fig_pathM, bbox_inches='tight', format='eps')
# # plt.close()
# # fig_pathM = 'OutletT.eps'
# # for i in range(1):
# # for i in range(0,len(DirList)):
#     plt.plot(sout[i,:,1], "--")
#     plt.plot(sout2[i,:,1], "--")
#     plt.plot(sout3[i,:,1], "--")
#     plt.plot(sout4[i,:,1], "--")
# # plt.yscale("log")
# # plt.xscale("log")
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(color="0.8", linewidth=0.5) 
# plt.ylabel("Temperature [K]", size=14)
# plt.xlabel("DoF", size=14)
# plt.legend(['Smooth surface x=0.1', 'Smooth surface x=0.2', 'Smooth surface x=0.3', 'Smooth surface x=0.4',
#             'A=0.001,W=4  x=0.1', 'A=0.001,W=4  x=0.2', 'A=0.001,W=4  x=0.3', 'A=0.001,W=4  x=0.4', 
#             'A=0.001,W=16  x=0.1', 'A=0.001,W=16  x=0.2', 'A=0.001,W=16  x=0.3', 'A=0.001,W=16  x=0.4', 
#             'A=0.002,W=4  x=0.1', 'A=0.002,W=4  x=0.2', 'A=0.002,W=4  x=0.3', 'A=0.002,W=4  x=0.4', 
#             'A=0.002,W=16  x=0.1', 'A=0.002,W=16  x=0.2', 'A=0.002,W=16  x=0.3', 'A=0.002,W=16  x=0.4'], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
# plt.savefig(fig_pathM, bbox_inches='tight', format='eps')
# plt.close()

QoIp00 = []
QoIT00 = []
QoIp14 = []
QoIp18 = []
QoIp16 = []
QoIp12 = []
QoIp24 = []
QoIp28 = []
QoIp26 = []
QoIp22 = []
QoIp34 = []
QoIp38 = []
QoIp36 = []
QoIp32 = []
QoIp44 = []
QoIp48 = []
QoIp46 = []
QoIp42 = []
QoIT14 = []
QoIT18 = []
QoIT16 = []
QoIT12 = []
QoIT24 = []
QoIT28 = []
QoIT26 = []
QoIT22 = []
QoIT34 = []
QoIT38 = []
QoIT36 = []
QoIT32 = []
QoIT44 = []
QoIT48 = []
QoIT46 = []
QoIT42 = []
xProf = [0.1, 0.2, 0.3, 0.4]
for i in range(0,len(DirList)):
    yPin = (sin[i,:,2] + 1.0/2.0 * sui[i,:,1] ** 2) * sui[i,:,1]
    yPout = (sout[i,:,2] + 1.0/2.0 * suo[i,:,1] ** 2) * suo[i,:,1]
    yTin = (Cp * sin[i,:,1]) * sui[i,:,1]
    yTout = (Cp * sout[i,:,1]) * suo[i,:,1]
    yPin2 = (sin2[i,:,2] + 1.0/2.0 * sui2[i,:,1] ** 2) * sui2[i,:,1]
    yPout2 = (sout2[i,:,2] + 1.0/2.0 * suo2[i,:,1] ** 2) * suo2[i,:,1]
    yTin2 = (Cp * sin2[i,:,1]) * sui2[i,:,1]
    yTout2 = (Cp * sout2[i,:,1]) * suo2[i,:,1]
    yPin3 = (sin3[i,:,2] + 1.0/2.0 * sui3[i,:,1] ** 2) * sui3[i,:,1]
    yPout3 = (sout3[i,:,2] + 1.0/2.0 * suo3[i,:,1] ** 2) * suo3[i,:,1]
    yTin3 = (Cp * sin3[i,:,1]) * sui3[i,:,1]
    yTout3 = (Cp * sout3[i,:,1]) * suo3[i,:,1]
    yPin4 = (sin4[i,:,2] + 1.0/2.0 * sui4[i,:,1] ** 2) * sui4[i,:,1]
    yPout4 = (sout4[i,:,2] + 1.0/2.0 * suo4[i,:,1] ** 2) * suo4[i,:,1]
    yTin4 = (Cp * sin4[i,:,1]) * sui4[i,:,1]
    yTout4 = (Cp * sout4[i,:,1]) * suo4[i,:,1]
    # IntpIn = simps(yPin, dx=0.00198)
    # IntTin = simps(yTin, dx=0.00198)
    # IntpOut = simps(yPout, dx=0.00198)
    # IntTout = simps(yTout, dx=0.00198)
    IntpIn = np.trapz(yPin, dx=0.00198)
    IntTin = np.trapz(yTin, dx=0.00198)
    IntpOut = np.trapz(yPout, dx=0.00198)
    IntTout = np.trapz(yTout, dx=0.00198)
    IntpIn2 = np.trapz(yPin2, dx=0.00198)
    IntTin2 = np.trapz(yTin2, dx=0.00198)
    IntpOut2 = np.trapz(yPout2, dx=0.00198)
    IntTout2 = np.trapz(yTout2, dx=0.00198)
    IntpIn3 = np.trapz(yPin3, dx=0.00198)
    IntTin3 = np.trapz(yTin3, dx=0.00198)
    IntpOut3 = np.trapz(yPout3, dx=0.00198)
    IntTout3 = np.trapz(yTout3, dx=0.00198)
    IntpIn4 = np.trapz(yPin4, dx=0.00198)
    IntTin4 = np.trapz(yTin4, dx=0.00198)
    IntpOut4 = np.trapz(yPout4, dx=0.00198)
    IntTout4 = np.trapz(yTout4, dx=0.00198)
    if i == 0:
        QoIp00.append(IntpIn-IntpOut)
        QoIT00.append(IntTout-IntTin)
        QoIp00.append(IntpIn2-IntpOut2)
        QoIT00.append(IntTout2-IntTin2)
        QoIp00.append(IntpIn3-IntpOut3)
        QoIT00.append(IntTout3-IntTin3)
        QoIp00.append(IntpIn4-IntpOut4)
        QoIT00.append(IntTout4-IntTin4)
        print(DirList[i])
    elif i ==1:
        QoIp14.append(IntpIn-IntpOut)
        QoIT14.append(IntTout-IntTin)
        QoIp14.append(IntpIn2-IntpOut2)
        QoIT14.append(IntTout2-IntTin2)
        QoIp14.append(IntpIn3-IntpOut3)
        QoIT14.append(IntTout3-IntTin3)
        QoIp14.append(IntpIn4-IntpOut4)
        QoIT14.append(IntTout4-IntTin4)
    elif i ==2:
        QoIp18.append(IntpIn-IntpOut)
        QoIT18.append(IntTout-IntTin)
        QoIp18.append(IntpIn2-IntpOut2)
        QoIT18.append(IntTout2-IntTin2)
        QoIp18.append(IntpIn3-IntpOut3)
        QoIT18.append(IntTout3-IntTin3)
        QoIp18.append(IntpIn4-IntpOut4)
        QoIT18.append(IntTout4-IntTin4)
    elif i ==3:
        QoIp16.append(IntpIn-IntpOut)
        QoIT16.append(IntTout-IntTin)
        QoIp16.append(IntpIn2-IntpOut2)
        QoIT16.append(IntTout2-IntTin2)
        QoIp16.append(IntpIn3-IntpOut3)
        QoIT16.append(IntTout3-IntTin3)
        QoIp16.append(IntpIn4-IntpOut4)
        QoIT16.append(IntTout4-IntTin4)
    elif i ==4:
        QoIp12.append(IntpIn-IntpOut)
        QoIT12.append(IntTout-IntTin)
        QoIp12.append(IntpIn2-IntpOut2)
        QoIT12.append(IntTout2-IntTin2)
        QoIp12.append(IntpIn3-IntpOut3)
        QoIT12.append(IntTout3-IntTin3)
        QoIp12.append(IntpIn4-IntpOut4)
        QoIT12.append(IntTout4-IntTin4)
    elif i ==5:
        QoIp24.append(IntpIn-IntpOut)
        QoIT24.append(IntTout-IntTin)
        QoIp24.append(IntpIn2-IntpOut2)
        QoIT24.append(IntTout2-IntTin2)
        QoIp24.append(IntpIn3-IntpOut3)
        QoIT24.append(IntTout3-IntTin3)
        QoIp24.append(IntpIn4-IntpOut4)
        QoIT24.append(IntTout4-IntTin4)
    elif i ==6:
        QoIp28.append(IntpIn-IntpOut)
        QoIT28.append(IntTout-IntTin)
        QoIp28.append(IntpIn2-IntpOut2)
        QoIT28.append(IntTout2-IntTin2)
        QoIp28.append(IntpIn3-IntpOut3)
        QoIT28.append(IntTout3-IntTin3)
        QoIp28.append(IntpIn4-IntpOut4)
        QoIT28.append(IntTout4-IntTin4)
    elif i ==7:
        QoIp26.append(IntpIn-IntpOut)
        QoIT26.append(IntTout-IntTin)
        QoIp26.append(IntpIn2-IntpOut2)
        QoIT26.append(IntTout2-IntTin2)
        QoIp26.append(IntpIn3-IntpOut3)
        QoIT26.append(IntTout3-IntTin3)
        QoIp26.append(IntpIn4-IntpOut4)
        QoIT26.append(IntTout4-IntTin4)
    elif i ==8:
        QoIp22.append(IntpIn-IntpOut)
        QoIT22.append(IntTout-IntTin)
        QoIp22.append(IntpIn2-IntpOut2)
        QoIT22.append(IntTout2-IntTin2)
        QoIp22.append(IntpIn3-IntpOut3)
        QoIT22.append(IntTout3-IntTin3)
        QoIp22.append(IntpIn4-IntpOut4)
        QoIT22.append(IntTout4-IntTin4)
    elif i ==9:
        QoIp34.append(IntpIn-IntpOut)
        QoIT34.append(IntTout-IntTin)
        QoIp34.append(IntpIn2-IntpOut2)
        QoIT34.append(IntTout2-IntTin2)
        QoIp34.append(IntpIn3-IntpOut3)
        QoIT34.append(IntTout3-IntTin3)
        QoIp34.append(IntpIn4-IntpOut4)
        QoIT34.append(IntTout4-IntTin4)
    elif i ==10:
        QoIp38.append(IntpIn-IntpOut)
        QoIT38.append(IntTout-IntTin)
        QoIp38.append(IntpIn2-IntpOut2)
        QoIT38.append(IntTout2-IntTin2)
        QoIp38.append(IntpIn3-IntpOut3)
        QoIT38.append(IntTout3-IntTin3)
        QoIp38.append(IntpIn4-IntpOut4)
        QoIT38.append(IntTout4-IntTin4)
    elif i ==11:
        QoIp36.append(IntpIn-IntpOut)
        QoIT36.append(IntTout-IntTin)
        QoIp36.append(IntpIn2-IntpOut2)
        QoIT36.append(IntTout2-IntTin2)
        QoIp36.append(IntpIn3-IntpOut3)
        QoIT36.append(IntTout3-IntTin3)
        QoIp36.append(IntpIn4-IntpOut4)
        QoIT36.append(IntTout4-IntTin4)
    elif i ==12:
        QoIp32.append(IntpIn-IntpOut)
        QoIT32.append(IntTout-IntTin)
        QoIp32.append(IntpIn2-IntpOut2)
        QoIT32.append(IntTout2-IntTin2)
        QoIp32.append(IntpIn3-IntpOut3)
        QoIT32.append(IntTout3-IntTin3)
        QoIp32.append(IntpIn4-IntpOut4)
        QoIT32.append(IntTout4-IntTin4)
    elif i ==13:
        QoIp44.append(IntpIn-IntpOut)
        QoIT44.append(IntTout-IntTin)
        QoIp44.append(IntpIn2-IntpOut2)
        QoIT44.append(IntTout2-IntTin2)
        QoIp44.append(IntpIn3-IntpOut3)
        QoIT44.append(IntTout3-IntTin3)
        QoIp44.append(IntpIn4-IntpOut4)
        QoIT44.append(IntTout4-IntTin4)
    elif i ==14:
        QoIp48.append(IntpIn-IntpOut)
        QoIT48.append(IntTout-IntTin)
        QoIp48.append(IntpIn2-IntpOut2)
        QoIT48.append(IntTout2-IntTin2)
        QoIp48.append(IntpIn3-IntpOut3)
        QoIT48.append(IntTout3-IntTin3)
        QoIp48.append(IntpIn4-IntpOut4)
        QoIT48.append(IntTout4-IntTin4)
    elif i ==15:
        QoIp46.append(IntpIn-IntpOut)
        QoIT46.append(IntTout-IntTin)
        QoIp46.append(IntpIn2-IntpOut2)
        QoIT46.append(IntTout2-IntTin2)
        QoIp46.append(IntpIn3-IntpOut3)
        QoIT46.append(IntTout3-IntTin3)
        QoIp46.append(IntpIn4-IntpOut4)
        QoIT46.append(IntTout4-IntTin4)
    elif i ==16:
        QoIp42.append(IntpIn-IntpOut)
        QoIT42.append(IntTout-IntTin)
        QoIp42.append(IntpIn2-IntpOut2)
        QoIT42.append(IntTout2-IntTin2)
        QoIp42.append(IntpIn3-IntpOut3)
        QoIT42.append(IntTout3-IntTin3)
        QoIp42.append(IntpIn4-IntpOut4)
        QoIT42.append(IntTout4-IntTin4)
    # Plot the L2 norm standardize error difference
    

fig_pathM = 'QoIp.eps'
plt.plot(xProf, QoIp00, "-o")
plt.plot(xProf, QoIp14, "-o")
# plt.plot(xProf, QoIp18, "-o")
# plt.plot(xProf, QoIp16, "-o")
# plt.plot(xProf, QoIp12, "-o")
# plt.plot(xProf, QoIp24, "-o")
# plt.plot(xProf, QoIp28, "-o")
plt.plot(xProf, QoIp26, "-o")
# plt.plot(xProf, QoIp22, "-o")
# plt.plot(xProf, QoIp34, "-.o")
# plt.plot(xProf, QoIp38, "-.o")
# plt.plot(xProf, QoIp36, "-.o")
# plt.plot(xProf, QoIp32, "-.o")
# plt.plot(xProf, QoIp44, "-.o")
# plt.plot(xProf, QoIp48, "-.o")
# plt.plot(xProf, QoIp46, "-.o")
plt.plot(xProf, QoIp42, "-.o")
# plt.yscale("log")
# plt.xscale("log")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(color="0.8", linewidth=0.5) 
plt.ylabel("Kinetic energy flux output", size=14)
plt.xlabel("x Profile", size=14)
# plt.legend(["A0-0_W0", "A0-001_W4", "A0-001_W8", "A0-001_W16", "A0-001_W32", "A0-002_W4", "A0-002_W8", "A0-002_W16", "A0-002_W32", "A0-003_W4", "A0-003_W8", "A0-003_W16", "A0-003_W32", "A0-004_W4", "A0-004_W8", "A0-004_W16", "A0-004_W32"], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
plt.legend(["A0-0_W0", "A0-001_W4", "A0-002_W16", "A0-004_W32"], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
plt.savefig(fig_pathM, bbox_inches='tight', format='eps')
plt.close()
# plt.show()
fig_pathM = 'QoIT.eps'
plt.plot(xProf, QoIT00, "-o")
# plt.plot(xProf, QoIT14, "-o")
# plt.plot(xProf, QoIT18, "-o")
# plt.plot(xProf, QoIT16, "-o")
# plt.plot(xProf, QoIT12, "-o")
# plt.plot(xProf, QoIT24, "-o")
# plt.plot(xProf, QoIT28, "-o")
# plt.plot(xProf, QoIT26, "-o")
# plt.plot(xProf, QoIT22, "-o")
# plt.plot(xProf, QoIT34, "-.o")
# plt.plot(xProf, QoIT38, "-.o")
# plt.plot(xProf, QoIT36, "-.o")
# plt.plot(xProf, QoIT32, "-.o")
# plt.plot(xProf, QoIT44, "-.o")
# plt.plot(xProf, QoIT48, "-.o")
# plt.plot(xProf, QoIT46, "-.o")
# plt.plot(xProf, QoIT42, "-.o")
# plt.yscale("log")
# plt.xscale("log")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(color="0.8", linewidth=0.5) 
plt.ylabel("Thermal power flux output", size=14)
plt.xlabel("x Profile", size=14)
# plt.legend(["A0-0_W0", "A0-001_W4", "A0-001_W8", "A0-001_W16", "A0-001_W32", "A0-002_W4", "A0-002_W8", "A0-002_W16", "A0-002_W32", "A0-003_W4", "A0-003_W8", "A0-003_W16", "A0-003_W32", "A0-004_W4", "A0-004_W8", "A0-004_W16", "A0-004_W32"], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
plt.legend(["A0-0_W0", "A0-001_W4", "A0-001_W16", "A0-004_W32"], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
plt.savefig(fig_pathM, bbox_inches='tight', format='eps')
plt.close()

#Calculate the quantity of interest, AKA the are under the curve for each case