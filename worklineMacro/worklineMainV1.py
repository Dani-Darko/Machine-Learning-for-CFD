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
import glob

# Call for all the subfunctions needed from the rest of the scripts
import ParametrizationWL
import SimulationWL
import SalomeRun
import OpenFOAMRun
import NeuralNetwork
# import QoIWL

# Path of this file and subsequently the OpenFOAM and NN data
path = os.getcwd()
path = path.replace('\\', '/')

print ("The current working directory is %s" % path)

# Call for the values that the ParametrizationWL script requires
currentxEl = ParametrizationWL.xElements
currentL = ParametrizationWL.L
# Input the amplitude and wavelength values by calling the ParametrizationWL function
Amp, WaveL, xArray, yArray, zArray0, zArray1, zArray05 = ParametrizationWL.CoorVals(currentxEl, currentL)

HPCr = input("Do you want to run the cases on HPC? y/n: " )
for i in range(0, len(Amp)):
    for j in range(0, len(WaveL)):
        # Create the new folders and updated files for each amplitude/wavelength
        # iteration calling the SimulationWL function
        SkipStep, NewPath = SimulationWL.CopyUpdate(currentxEl, Amp[i], WaveL[j], xArray, yArray, zArray0, zArray1, zArray05)
        # If the folder was created anew, then run the rest of the steps to built
        # the simulation data for this case. If not, go to the next Amp and WaveL
        if SkipStep != "yes":
            # Run Salome and generate the STL file
            SalomeRun.RunSalome(Amp[i], WaveL[j], currentxEl, NewPath)
            # Run locally or on HPC
            if HPCr == "n":
                # Reload the OpenFOAM module after updating the script from the previous step
                OpenFOAMRun = reload(OpenFOAMRun)
                
                # Run OpenFOAM simulations for each newly created testcase
                OpenFOAMRun.RunOpenFOAM(Amp[i], WaveL[j])
if HPCr == "y":
    # Prompt the command line to run in HPC with all all the files needed
    DirList = [os.path.basename(x) for x in glob.glob("../caseDatabase/*")]
    print("Copy the /caseDatabase directory into your HPC system")
    print("Inside the /caseDatabase directory, run with the command and jobnames: sbatch -a", ', '.join(DirList), "./HPCjobArray.sh")
    input("Once the files are updated locally, press 'ENTER' to continue...")

# Create the tensors from the old & new database for the NN
epochs = int(input("Number of epoch for the Networks training: "))
x, stanBCList, stanBCListNames = NeuralNetwork.DataDoF(path)
for stanBCi in range(0,len(stanBCList)):
    # Create the features matrix, calculate the SVD and standardize the data
    modes, x_train, x_valid, stanBC_train, stanBC_valid, Tr_train, Tr_valid = NeuralNetwork.DataSVD(x, stanBCList[i], stanBCListNames[stanBCi])
    
    # Run the neural network training update
    NeuralNetwork.UpdateParametersSVD(stanBCListNames[stanBCi], epochs, modes, x_train, x_valid, Tr_train, Tr_valid)
    NeuralNetwork.UpdateParametersDoF(stanBCListNames[stanBCi], epochs, x_train, x_valid, stanBC_train, stanBC_valid)
    # NeuralNetwork.UpdateParametersRBF(stanBCListNames[stanBCi], epochs, modes, x_train, x_valid,Tr_train, Tr_valid)
    
print("En hora buena! Neural network parameters properly updated!")