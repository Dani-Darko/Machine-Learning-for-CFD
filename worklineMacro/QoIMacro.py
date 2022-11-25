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
# import SalomeRun
import OpenFOAMRun
import NeuralNetwork
import QoI

# Path of this file and subsequently the OpenFOAM and NN data
path = os.getcwd()
path = path.replace('\\', '/')

print ("The current working directory is %s" % path)

# Create the tensors from the old & new database for the NN
x, stanBCList, stanBCListNames, BCtensorList, xsd0, xmean0, xsd1, xmean1 = NeuralNetwork.DataDoF(path)
QP, QT = QoI.QoIWL(x, BCtensorList, stanBCListNames, xsd0, xmean0, xsd1, xmean1)