"""
1. ParametrizationWL.py takes geometry parameter values from 
    the user to input into functions and built the points coordinates
    for the OpenFOAM files

@author: Daniela Segura
"""

# # Numpy libraries for array and matrix calculations
import numpy as np
from numpy import array
from numpy import diag
from numpy import zeros

# Constants
xElements = 300
L = 2.0

# Create the Parametrization function for worklineMainV1 to call
def CoorVals(xElements, L):
    # Input the amount of amplitude and wavelength values as a user
    AmpLength = int(input("Enter the amount of amplitude values you need: " ))
    WaveLLength = int(input("Enter the amount of wavelength values you need: " ))
    AmpArray = []
    print("Enter the Amplitude values: ")
    # Input Amp values till the max amount of Amplitude values is reached
    for i in range(0, AmpLength):
        Aele = float(input()) # New value
        AmpArray.append(Aele) # Add to list of Amp
    WaveLArray = []
    print("Enter the Wavelength values: ")
    # Input WaveL values till the max amount of Wavelength values is reached
    for i in range(0, WaveLLength):
        Wele = float(input()) # New value
        WaveLArray.append(Wele) # Add to list of WaveL
    # Transform lists to arrays
    Amp = np.asarray(AmpArray)
    WaveL = np.asarray(WaveLArray)
    # Create the coordinate arrays to construct the matrices in the files
    # blockMeshDict and Sample
    xStep = L/xElements
    xArray = np.arange(0, L+xStep, xStep)
    for i in range(0,AmpLength):
        for j in range(0,WaveLLength):
            yArray = -AmpArray[i]*np.cos(xArray*WaveLArray[j]*np.pi)+0.2+AmpArray[i]
    zArray0 = 0.0
    zArray1 = 0.1
    zArray05 = 0.05
    # Share the relevant arrays with the parent script
    return Amp, WaveL, xArray, yArray, zArray0, zArray1, zArray05
