"""
Workline macro for process control automatization.

This script calls for other 4. Each script has a particular job to
built up the simulation database. These script files are:
    1. ParametrizationWL.py takes geometry parameter values from 
    the user to input into functions and built the points coordinates
    for the OpenFOAM files
    2. SimulationWL.py copies the base OpenFOAM folder and files
    into a new folder named after the relevant parameters taken from
    step 1. while editing the relevant data inside blockMeshDict
    and Sample to run the simulations with the proper geometry from Salome
    3. NNDecoupledSVD.py runs the Neural Network training update
    by adding the simulations from step 2. into the set, this is
    updated in file, so the new values stay for future runs
    4. QoIWL.py calculates the heat and kinetic energy fluxes from the
    Neural Network predictions or if validation is needed, from the
    simulation data.

The software used is OpenFOAM v2106 for the high fidelity simulations,
and Salome 9.7.0 to generate the geometry to set the topology

@author: Daniela Segura
"""

# System, path and runtime information
import os
from importlib import reload
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

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

# Call for the values that the ParametrizationWL script requires
currentxEl = ParametrizationWL.xElements
currentL = ParametrizationWL.L
# Choose between cosine function or nerves spline
# Param = "Cosine"
# Param = "Nerves"
# Param = "DoubleCosine"
Param = "none"

if Param == "Cosine":
    # Input the amplitude and wavelength values by calling the ParametrizationWL function
    Amp, WaveL, xArray, zArray0, zArray1, zArray05 = ParametrizationWL.CoorVals(currentxEl, currentL)
    for i in range(0, len(Amp)):
        for j in range(0, len(WaveL)):
            # Create the new folders and updated files for each amplitude/wavelength
            # iteration calling the SimulationWL function
            SimulationWL = reload(SimulationWL)
            SkipStep, NewPath = SimulationWL.CopyUpdate(currentxEl, Amp[i], WaveL[j], xArray, zArray0, zArray1, zArray05)
            # If the folder was created anew, then run the rest of the steps to built
            # the simulation data for this case. If not, go to the next Amp and WaveL
            if SkipStep != "yes":
                # Reload the OpenFOAM and Salome modules after updating the script from the previous step
                # SalomeRun = reload(SalomeRun)
                OpenFOAMRun = reload(OpenFOAMRun)
                # Run the mesh generation before copiing the cellZone from the original mesh
                OpenFOAMRun.RunMesh(NewPath)
                # Run Salome and generate the STL file
                # SalomeRun.RunSalome(Amp[i], WaveL[j], currentxEl, NewPath)
                # Run OpenFOAM simulations for each newly created testcase
                # OpenFOAMRun.RunOpenFOAM()
if Param == "DoubleCosine":
    # Input the amplitude and wavelength values by calling the ParametrizationWL function
    Amp, WaveL, xArray, zArray0, zArray1, zArray05 = ParametrizationWL.CoorValsMultiAW(currentxEl, currentL, path)
    for i in range(0, len(Amp)):
        for j in range(0, len(WaveL)):
            # Create the new folders and updated files for each amplitude/wavelength
            # iteration calling the SimulationWL function
            SimulationWL = reload(SimulationWL)
            SkipStep, NewPath = SimulationWL.CopyUpdateMultiAW(currentxEl, Amp[i,0], WaveL[j,0], Amp[i,1], WaveL[j,1], xArray, zArray0, zArray1, zArray05)
            # If the folder was created anew, then run the rest of the steps to built
            # the simulation data for this case. If not, go to the next Amp and WaveL
            if SkipStep != "yes":
                # Reload the OpenFOAM and Salome modules after updating the script from the previous step
                # SalomeRun = reload(SalomeRun)
                OpenFOAMRun = reload(OpenFOAMRun)
                # Run the mesh generation before copiing the cellZone from the original mesh
                OpenFOAMRun.RunMesh(NewPath)
                # Run Salome and generate the STL file
                # SalomeRun.RunSalome(Amp[i], WaveL[j], currentxEl, NewPath)
                # Run OpenFOAM simulations for each newly created testcase
                # OpenFOAMRun.RunOpenFOAM()
elif Param == "Nerves":
    # Input the main nerves coordinate values by calling the ParametrizationWL function
    xCn, yCn, xArray, zArray0, zArray1, zArray05 = ParametrizationWL.CoorValsNerves(currentxEl, currentL, path)
    for i in range(0, len(xCn)):
        for j in range(0, len(yCn)):
            # Create the new folders and updated files for each amplitude/wavelength
            # iteration calling the SimulationWL function
            SimulationWL = reload(SimulationWL)
            SkipStep, NewPath = SimulationWL.CopyUpdateNerves(currentxEl, currentL, xCn[i], yCn[j], xArray, zArray0, zArray1, zArray05)
            # If the folder was created anew, then run the rest of the steps to built
            # the simulation data for this case. If not, go to the next Amp and WaveL
            if SkipStep != "yes":
                # Reload the OpenFOAM and Salome modules after updating the script from the previous step
                # SalomeRun = reload(SalomeRun)
                OpenFOAMRun = reload(OpenFOAMRun)
                # Run the mesh generation before copiing the cellZone from the original mesh
                OpenFOAMRun.RunMesh(NewPath)
                # Run Salome and generate the STL file
                # SalomeRun.RunSalome(Amp[i], WaveL[j], currentxEl, NewPath)
                # Run locally or on HPC
# Create a list with both old & new data sampling
# OpenFOAMRun.RunOpenFOAMHPC(path)

newxin = np.loadtxt("Xnew.txt", delimiter=',')
# # stanBCList = [stanui, stanuo, stanTi, stanTo, stanpi, stanpo]
x, stanBCList, stanBCListNames, BCtensorList, xsd, xmean, feat = NeuralNetwork.DataQoI2(newxin[:,:2], path)
newx = np.empty([len(newxin[:,0]),len(newxin[0,:2])])
for xj in range(len(newxin[0,:2])):
    for xi in range(len(newxin[:,0])):
        newx[xi,xj] = (newxin[xi,xj]-xmean[xj])/xsd[xj]
newx = torch.from_numpy(newx)

# Create the tensors from the old & new database for the NN
# x, stanBCList, stanBCListNames, BCtensorList, xsd, xmean, feat = NeuralNetwork.DataDoF(path)
QP, QT = QoI.QoIWL(x, BCtensorList, stanBCListNames, xsd, xmean, feat)

# How many iterations the neural networks will use to train
# epochs = int(input("Number of epoch for the Networks training: "))
epochs = 20000
print("Number of epoch for the Networks training: ", epochs)

for stanBCi in range(0,len(stanBCList)):
    # Create the features matrix, calculate the SVD and standardize the data
    modes, x_train, x_valid, stanBC_train, stanBC_valid, Tr_train, Tr_valid = NeuralNetwork.DataSVD(x, stanBCList[stanBCi], stanBCListNames[stanBCi])
    
    # Run the neural network training update
    NeuralNetwork.UpdateParametersSVD(feat, stanBCListNames[stanBCi], epochs, modes, x_train, x_valid, Tr_train, Tr_valid)
    NeuralNetwork.UpdateParametersDoF(feat, stanBCListNames[stanBCi], epochs, x_train, x_valid, stanBC_train, stanBC_valid)
    NeuralNetwork.UpdateParametersRBF(stanBCListNames[stanBCi], modes, x_train, x_valid, Tr_train, Tr_valid)
    NeuralNetwork.UpdateParametersKriging(stanBCListNames[stanBCi], modes, x_train, x_valid, Tr_train, Tr_valid)
NeuralNetwork.UpdateParametersQoI(feat, QP, epochs, x_train, x_valid, "Pressure")
NeuralNetwork.UpdateParametersQoI(feat, QT, epochs, x_train, x_valid, "Temperature")
    
print("En hora buena! Neural network parameters properly updated!")

# Import the features data from a file
SVDp = []
DOFp = []
RBFp = []
KQp = []
newxin = np.loadtxt("Xnew.txt", delimiter=',')
newx = np.empty([len(newxin[:,0]),len(newxin[0,:2])])
for xj in range(len(newxin[0,:2])):
    for xi in range(len(newxin[:,0])):
        newx[xi,xj] = (newxin[xi,xj]-xmean[xj])/xsd[xj]
newx = torch.from_numpy(newx)
# # stanBCList = [stanui, stanuo, stanTi, stanTo, stanpi, stanpo]
hidden_sizes = 32
print("Calculating predictions of each surrogate model...")
std = np.loadtxt('stdData.txt')
mean = np.loadtxt('meanData.txt')
for stanBCi in range(0,len(stanBCList)):
    SVDp.append(PredictionWL.PredsSVD(newx, stanBCListNames[stanBCi], hidden_sizes, feat, modes, xsd, xmean, std[stanBCi], mean[stanBCi]))
    DOFp.append(PredictionWL.PredsDoF(newx, stanBCListNames[stanBCi], hidden_sizes, feat, len(BCtensorList[4][0,:]), xsd, xmean, std[stanBCi], mean[stanBCi]))
    RBFp.append(PredictionWL.PredsRBF(newx, stanBCListNames[stanBCi], xsd, xmean, modes, std[stanBCi], mean[stanBCi]))
    KQp.append(PredictionWL.PredsKriging(newx, stanBCListNames[stanBCi], xsd, xmean, modes, std[stanBCi], mean[stanBCi]))
NNQP = PredictionWL.PredsQoI(newx, "Pressure", hidden_sizes, feat, 1, xmean, xsd)
NNQT = PredictionWL.PredsQoI(newx, "Temperature", hidden_sizes, feat, 1, xmean, xsd)
xlists =[]
for i in range(2):
    xlist = list(dict.fromkeys(newxin[1:,i]))
    xlists.append(xlist)
X, Y = np.meshgrid(xlists[0], xlists[1])
stanBCList1, stanBCListNames1, BCtensorList1 = NeuralNetwork.DataQoI(newxin, path)
newQP, newQT = QoI.QoIWL(newx, BCtensorList1, stanBCListNames1, xsd, xmean, feat)

fig_pathProf = '../SVDdata/QoI/TprofilePreds.eps'
plt.plot(SVDp[3][-2,:])
plt.plot(DOFp[3][-2,:])
plt.plot(RBFp[3][-2,:])
plt.plot(KQp[3][-2,:])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(color="0.8", linewidth=0.5) 
# plt.xlabel("Rawdatapoint", size=14)
plt.ylabel("Temperature [K]", size=14)
plt.savefig(fig_pathProf, bbox_inches='tight', format='eps')
plt.close()

print("Calculating quantities of interest...")
# Calculate the quantity of interest
gridQP, gridQT = QoI.gridQoI(xlists, newxin[:,0:2], newQP, newQT, xsd, xmean)
gridSVDQP, gridSVDQT, SVDQP, SVDQT = QoI.calcQoI(xlists, newx[:,0:2], newxin[:,0:2], SVDp, xsd, xmean)
gridDOFQP, gridDOFQT, DOFQP, DOFQT = QoI.calcQoI(xlists, newx[:,0:2], newxin[:,0:2], DOFp, xsd, xmean)
gridRBFQP, gridRBFQT, RBFQP, RBFQT = QoI.calcQoI(xlists, newx[:,0:2], newxin[:,0:2], RBFp, xsd, xmean)
gridKQP, gridKQT, KQP, KQT = QoI.calcQoI(xlists, newx[:,0:2], newxin[:,0:2], KQp, xsd, xmean)
gridNNQP, gridNNQT = QoI.gridQoI(xlists, newxin[:,0:2], NNQP, NNQT, xsd, xmean)
gridQPp = [gridQP, gridSVDQP, gridDOFQP, gridRBFQP, gridNNQP, gridKQP]
gridQTp = [gridQT, gridSVDQT, gridDOFQT, gridRBFQT, gridNNQT, gridKQT]
QPp = [newQP, SVDQP, DOFQP, RBFQP, NNQP, KQP]
QTp = [newQT, SVDQT, DOFQT, RBFQT, NNQT, KQT]
contName = ["Sims","SVD", "DOF", "RBF", "NNQ", "Kriging"]

wpars = np.arange(0,1.25,0.25)
for Qi in range(6):
    for wpar in wpars:
        cont = wpar*gridQTp[Qi] - (1 - wpar)*gridQPp[Qi]
        scat = wpar*QTp[Qi] - (1 - wpar)*QPp[Qi]
        fig_path = '../SVDdata/cont'+contName[Qi]+'_'+str(wpar).replace('.', '-')+'.eps'
        fig, ax = plt.subplots()
        # surf = ax.contour(X, Y, cont, cmap=cm.jet, linewidth=0.1)
        surf = ax.contourf(X, Y, cont, cmap=cm.jet)
        # ax.scatter(newxin[1:,0], newxin[1:,1], scat[1:])
        plt.plot(newxin[1:,0], newxin[1:,1], 'k.')
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax.set_xlabel("Amplitude", size=14)
        ax.set_ylabel("Wavelength", size=14)
        ax.set_title(f"Thermohydraulic energy flux for w = {wpar}", size=14)
        plt.savefig(fig_path, bbox_inches='tight', format='eps')
        plt.close()