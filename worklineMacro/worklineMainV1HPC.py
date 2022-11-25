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
print("hello")
# System, path and runtime information
import os
from importlib import reload
import time

# Call for all the subfunctions needed from the rest of the scripts
import ParametrizationWLHPC
import SimulationWLHPC
# import SalomeRun
import OpenFOAMRunHPC
import NeuralNetwork
import QoI

# Path of this file and subsequently the OpenFOAM and NN data
path = os.getcwd()
path = path.replace('\\', '/')

print ("The current working directory is %s" % path)

# Call for the values that the ParametrizationWL script requires
currentxEl = ParametrizationWLHPC.xElements
currentL = ParametrizationWLHPC.L
# Choose between cosine function or nerves spline
# Param = "Cosine"
# Param = "Nerves"
Param = "DoubleCosine"

if Param == "Cosine":
    # Input the amplitude and wavelength values by calling the ParametrizationWL function
    Amp, WaveL, xArray, zArray0, zArray1, zArray05 = ParametrizationWLHPC.CoorVals(currentxEl, currentL)
    for i in range(0, len(Amp)):
        for j in range(0, len(WaveL)):
            # Create the new folders and updated files for each amplitude/wavelength
            # iteration calling the SimulationWL function
            SimulationWLHPC = reload(SimulationWLHPC)
            SkipStep, NewPath = SimulationWLHPC.CopyUpdate(currentxEl, Amp[i], WaveL[j], xArray, zArray0, zArray1, zArray05)
            # If the folder was created anew, then run the rest of the steps to built
            # the simulation data for this case. If not, go to the next Amp and WaveL
            if SkipStep != "yes":
                # Reload the OpenFOAM and Salome modules after updating the script from the previous step
                # SalomeRun = reload(SalomeRun)
                OpenFOAMRunHPC = reload(OpenFOAMRunHPC)
                # Run the mesh generation before copiing the cellZone from the original mesh
                OpenFOAMRunHPC.RunMesh(NewPath)
                # Run Salome and generate the STL file
                # SalomeRun.RunSalome(Amp[i], WaveL[j], currentxEl, NewPath)
                # Run on HPC
                # Run OpenFOAM simulations for each newly created testcase
                # OpenFOAMRunHPC.RunOpenFOAM()
if Param == "DoubleCosine":
    # Input the amplitude and wavelength values by calling the ParametrizationWL function
    Amp, WaveL, xArray, zArray0, zArray1, zArray05 = ParametrizationWLHPC.CoorValsMultiAW(currentxEl, currentL, path)
    for i in range(0, len(Amp)):
        for j in range(0, len(WaveL)):
            # Create the new folders and updated files for each amplitude/wavelength
            # iteration calling the SimulationWL function
            SimulationWLHPC = reload(SimulationWLHPC)
            SkipStep, NewPath = SimulationWLHPC.CopyUpdateMultiAW(currentxEl, Amp[i,0], WaveL[j,0], Amp[i,1], WaveL[j,1], xArray, zArray0, zArray1, zArray05)
            # If the folder was created anew, then run the rest of the steps to built
            # the simulation data for this case. If not, go to the next Amp and WaveL
            if SkipStep != "yes":
                # Reload the OpenFOAM and Salome modules after updating the script from the previous step
                # SalomeRun = reload(SalomeRun)
                OpenFOAMRunHPC = reload(OpenFOAMRunHPC)
                # Run the mesh generation before copiing the cellZone from the original mesh
                OpenFOAMRunHPC.RunMesh(NewPath)
                # Run Salome and generate the STL file
                # SalomeRun.RunSalome(Amp[i], WaveL[j], currentxEl, NewPath)
                # Run on HPC
                # Run OpenFOAM simulations for each newly created testcase
                # OpenFOAMRunHPC.RunOpenFOAM()
elif Param == "Nerves":
    # Input the main nerves coordinate values by calling the ParametrizationWL function
    xCn, yCn, xArray, zArray0, zArray1, zArray05 = ParametrizationWLHPC.CoorValsNerves(currentxEl, currentL, path)
    for i in range(0, len(xCn)):
        for j in range(0, len(yCn)):
            # Create the new folders and updated files for each amplitude/wavelength
            # iteration calling the SimulationWL function
            SimulationWLHPC = reload(SimulationWLHPC)
            SkipStep, NewPath = SimulationWLHPC.CopyUpdateNerves(currentxEl, currentL, xCn[i], yCn[j], xArray, zArray0, zArray1, zArray05)
            # If the folder was created anew, then run the rest of the steps to built
            # the simulation data for this case. If not, go to the next Amp and WaveL
            if SkipStep != "yes":
                # Reload the OpenFOAM and Salome modules after updating the script from the previous step
                # SalomeRun = reload(SalomeRun)
                OpenFOAMRunHPC = reload(OpenFOAMRunHPC)
                # Run the mesh generation before copiing the cellZone from the original mesh
                OpenFOAMRunHPC.RunMesh(NewPath)
                # Run Salome and generate the STL file
                # SalomeRun.RunSalome(Amp[i], WaveL[j], currentxEl, NewPath)
                # Run on HPC
                # Run OpenFOAM simulations for each newly created testcase
OpenFOAMRunHPC.RunOpenFOAMHPC(path)

# Create the tensors from the old & new database for the NN
# time.sleep(300)
# x, stanBCList, stanBCListNames, BCtensorList, xsd, xmean, feat = NeuralNetwork.DataDoF(path)
# QP, QT = QoI.QoIWL(x, BCtensorList, stanBCListNames, xsd, xmean)

# # How many iterations the neural networks will use to train
# epochs = 10000
# print("Number of epoch for the Networks training: ", epochs)

# for stanBCi in range(0,len(stanBCList)):
#     # Create the features matrix, calculate the SVD and standardize the data
#     modes, x_train, x_valid, stanBC_train, stanBC_valid, Tr_train, Tr_valid = NeuralNetwork.DataSVD(x, stanBCList[i], stanBCListNames[stanBCi])
    
#     # Run the neural network training update
#     NeuralNetwork.UpdateParametersSVD(feat, stanBCListNames[stanBCi], epochs, modes, x_train, x_valid, Tr_train, Tr_valid)
#     NeuralNetwork.UpdateParametersDoF(feat, stanBCListNames[stanBCi], epochs, x_train, x_valid, stanBC_train, stanBC_valid)
#     # NeuralNetwork.UpdateParametersRBF(stanBCListNames[stanBCi], modes, x_train, x_valid, Tr_train, Tr_valid)
# NeuralNetwork.UpdateParametersQoI(feat, QP, epochs, x_train, x_valid, "Pressure")
# NeuralNetwork.UpdateParametersQoI(feat, QT, epochs, x_train, x_valid, "Temperature")
    
# print("En hora buena! Neural network parameters properly updated!")

