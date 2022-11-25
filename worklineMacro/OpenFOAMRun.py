"""
2. OpenFoamRun.py Runs the batch files within HPC or the OpenFOAM files in
    local to create the mesh and run the simulations
@author: Daniela Segura
"""
# Subprocess and time libraries to run terminal commands and software
import subprocess
import shutil

def RunOpenFOAM():
    # Make a subprocess call to the openfoam2106 terminal and run the relevant ./Allrun
    subprocess.call(['openfoam2106 ./../caseDatabase/A0-005_0-005_W16-0_24-0/Allrun'], shell=True)
def RunPreproc():
    # Make a subprocess call to the openfoam2106 terminal and run the relevant ./Allrun
    subprocess.call(['openfoam2106 ./../caseDatabase/A0-005_0-005_W16-0_24-0/Preproc'], shell=True)
def RunMesh(NewPath):
    # Make a subprocess call to the openfoam2106 terminal and run the relevant ./Allrun
    subprocess.call(['openfoam2106 ./../caseDatabase/A0-005_0-005_W16-0_24-0/runMesh'], shell=True)
    # Move the cellZones file to the corresponding folder
    shutil.move('/media/dani/Data/Ubuntufiles/ProperThermoProp/caseDatabase/A0-005_0-005_W16-0_24-0/cellZones', NewPath + '/constant/polyMesh/cellZones')
    print("cellZones succesfully copied to constant/polyMesh")
import glob
import os
def RunOpenFOAMHPC(path):
    # Create a list with both old & new data sampling
    DirList1 = glob.glob("../caseDatabase/*")
    DirList = []
    for Dir1 in DirList1:
        # Make a subprocess call to the openfoam2106 terminal and run the relevant ./Allrun
        DirList1 = glob.glob("../caseDatabase/*")
        if not os.path.exists(Dir1 + '/log.chtMultiRegionSimpleFoam'):
            DirList.append(Dir1)
    for Diri in DirList:
        fileFold = './'+Diri+'/Allrun'
        subprocess.call([f"openfoam2106 {fileFold}"], shell=True)