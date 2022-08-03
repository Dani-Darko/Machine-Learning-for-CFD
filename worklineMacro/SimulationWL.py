"""
2. SimulationWL.py copies the base OpenFOAM folder and files
    into a new folder named after the relevant parameters taken from
    step 1. while editing the relevant data inside blockMeshDict
    and Sample to run the simulations with the proper geometry

@author: Daniela Segura
"""
# System, path and runtime information
import os
import shutil

# Create the Parametrization function for worklineMainV1 to call
def CopyUpdate(xElements, Amp, WaveL, xArray, yArray, zArray0, zArray1, zArray05):
    # Copy the files from the base case to a new folder with "Amp" and "WaveL" as name
    SourceDir = 'baseHFMFiles'
    # Transform the float values of amplitude and wavelength into string to
    # name the folders
    Amp = str(Amp).replace('.', '-')
    WaveL = str(int(WaveL))
    NewDirectory = "A"+Amp+"_W"+WaveL
    ParentDir = '/media/dani/Data/Ubuntufiles/ProperThermoProp/caseDatabase/'
    NewPath = os.path.join(ParentDir, NewDirectory)
    #if path already exists, remove it before copying with copytree()
    if os.path.exists(NewPath):
        print("The folder '% s' already exists, no further action needed..." % NewDirectory)
        SkipStep = "yes"
        return SkipStep, NewPath
    else:
        shutil.copytree(SourceDir, NewPath)
        print("Simulation files successfully copied to directory '% s'..." % NewDirectory)
        
        # Write a file with the Amp and WaveL data for future reference within the NN
        AWfile = open("../caseDatabase/"+NewDirectory+"/AmpWaveL.txt", "w")
        AWfile.write(str(Amp).replace('-','.')+"\n")
        AWfile.write(str(WaveL)+"\n")
        
        # Open the blockMeshDict and Sample to read the initial data the matrix data
        meshFilePath = NewPath + '/system/blockMeshDict'
        meshFile = open(meshFilePath)
        meshFileCont = meshFile.readlines()
        meshFile.close()
        sampleFilePath = NewPath + '/system/sample'
        sampleFile = open(sampleFilePath)
        sampleFileCont = sampleFile.readlines()
        sampleFile.close()
        
        # Compose the matrix data to write in the files
        meshFileCont.insert(46, 'polyLine 3 2\n')
        meshFileCont.insert(47, '(\n')
        for i in range(48,48+xElements+1):
            CoorListElement0 = "( " + str(xArray[i-48]) + " " + str(yArray[i-48]) + " " + str(zArray0) + " )\n"
            meshFileCont.insert(i, CoorListElement0)
        meshFileCont.insert(i+1, ')\n')
        meshFileCont.insert(i+2, 'polyLine 7 6\n')
        meshFileCont.insert(i+3, '(\n')
        j = i+4
        for i in range(j,j+xElements+1):
            CoorListElement1 = "( " + str(xArray[i-j]) + " " + str(yArray[i-j]) + " " + str(zArray1) + " )\n"
            meshFileCont.insert(i, CoorListElement1)
        meshFileCont.insert(i+1, ')\n')
        meshFileCont.insert(i+2, 'polyLine 8 9\n')
        meshFileCont.insert(i+3, '(\n')
        j = i+4
        for i in range(j,j+xElements+1):
            CoorListElement0 = "( " + str(xArray[i-j]) + " " + str(yArray[i-j]) + " " + str(zArray0) + " )\n"
            meshFileCont.insert(i, CoorListElement0)
        meshFileCont.insert(i+1, ')\n')
        meshFileCont.insert(i+2, 'polyLine 12 13\n')
        meshFileCont.insert(i+3, '(\n')
        j = i+4
        for i in range(j,j+xElements+1):
            CoorListElement1 = "( " + str(xArray[i-j]) + " " + str(yArray[i-j]) + " " + str(zArray1) + " )\n"
            meshFileCont.insert(i, CoorListElement1)
        meshFileCont.insert(i+1, ')\n')
        
        for i in range(65,65+xElements+1):
            CoorListElement05 = "( " + str(xArray[i-65]) + " " + str(yArray[i-65]-0.0002) + " " + str(zArray05) + " )\n"
            sampleFileCont.insert(i, CoorListElement05)
    
        # Open the files again to overwrite with the updated data
        meshFile = open(meshFilePath, "w")
        newMeshFileCont = "".join(meshFileCont)
        meshFile.write(newMeshFileCont)
        meshFile.close()
        sampleFile = open(sampleFilePath, "w")
        newSampleFileCont = "".join(sampleFileCont)
        sampleFile.write(newSampleFileCont)
        sampleFile.close()
        print("Files 'blockMeshDict' and 'sample' successfully updated with amplitud and wavelength data...")
    
        # Update the OpenFOAMRun.py script to run each new case
        OFfile = open("OpenFOAMRun.py")
        OFstring = OFfile.readlines()
        OFfile.close()
        OFstring[10] ="    subprocess.call(['openfoam2106 ./../caseDatabase/"+NewDirectory+"/Allrun'], shell=True)"
        OFfile = open("OpenFOAMRun.py", "w")
        newOFcont = "".join(OFstring)
        OFfile.write(newOFcont)
        OFfile.close()
        # Make a subpro
        
        # When this script is finished, no data is required
        SkipStep = "no"
        return SkipStep, NewPath