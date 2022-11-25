"""
2. OpenFoamRun.py Runs the batch files within HPC or the OpenFOAM files in
    local to create the mesh and run the simulations.
@author: Daniela Segura
"""
# Subprocess and time libraries to run terminal commands and software
import subprocess
import shutil
import time
import os
def RunOpenFOAM():
    # Make a subprocess call to the openfoam2106 terminal and run the relevant ./Allrun
    return_code = subprocess.Popen('sbatch ./../caseDatabase/A0-0_0-0_W0-0_0-0/Allrun.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
def RunPreproc():
    # Make a subprocess call to the openfoam2106 terminal and run the relevant ./Preprocess
    return_code = subprocess.Popen('sbatch ./../caseDatabase/A0-0_0-0_W0-0_0-0/Preproc.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
def RunMesh(NewPath):
    # Make a subprocess call to the openfoam2106 terminal and run the relevant ./runMesh
    subprocess.Popen(['sbatch', './../caseDatabase/A0-0_0-0_W0-0_0-0/runMesh.sh'])
    # Move the cellZones file to the corresponding folder
    time.sleep(30)
    # Move the cellZones file to the corresponding folder
    shutil.move('/lustrehome/home/s.2115589/caseDatabase/A0-0_0-0_W0-0_0-0/cellZones', NewPath + '/constant/polyMesh/cellZones')
import glob
def RunOpenFOAMHPC(path):
    # Create a list with both old & new data sampling
    DirList = glob.glob("../caseDatabase/*")
    for Diri in DirList:
        # Make a subprocess call to the openfoam2106 terminal and run the relevant ./Allrun
        fileFold = './'+Diri+'/Allrun.sh'
        subprocess.Popen([f"sbatch {fileFold}"], shell=True)
        