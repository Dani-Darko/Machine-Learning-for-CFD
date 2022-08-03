"""
2. SalomeRun.py Opens Salome to run the SalomeGeom.py script and build the
    geometry used for topology application on the Helium and Solid regions.
@author: Daniela Segura
"""
# Subprocess and time libraries to run terminal commands and software
import subprocess

def RunOpenFOAM(Amp, WaveL):
    # Make a subprocess call to the openfoam2106 terminal and run the relevant ./Allrun
    subprocess.call(['openfoam2106 ./../caseDatabase/A0-002_W22/Allrun'], shell=True)