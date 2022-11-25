"""
2. SimulationWL.py copies the base OpenFOAM folder and files
    into a new folder named after the relevant parameters taken from
    step 1. while editing the relevant data inside blockMeshDict
    and Sample to run the simulations with the proper geometry

@author: Daniela Segura
"""
# System, path and runtime information
import glob
import os
import shutil
import numpy as np
from scipy.interpolate import CubicSpline

# Create the Parametrization function for worklineMainV1 to call
def CopyUpdate(xElements, Amp, WaveL, xArray, zArray0, zArray1, zArray05):
    # Create the yArray
    yArray = -Amp*np.cos(xArray*WaveL*np.pi)+0.2+Amp
    # Copy the files from the base case to a new folder with "Amp" and "WaveL" as name
    SourceDir = 'baseHFMFiles'
    # Transform the float values of amplitude and wavelength into string to
    # name the folders
    Amp = str(Amp).replace('.', '-')
    WaveL = str(WaveL).replace('.', '-')
    NewDirectory = "A"+Amp+"_W"+WaveL
    ParentDir = '/lustrehome/home/s.2115589/caseDatabase/'
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
        AWfile.write(str(WaveL).replace('-','.')+"\n")
        
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
        fileCountStart = 56
        meshFileCont.insert(fileCountStart, 'polyLine 3 2\n')
        meshFileCont.insert(fileCountStart+1, '(\n')
        for i in range(fileCountStart+2,fileCountStart+xElements+3):
            CoorListElement0 = "( " + str(xArray[i-fileCountStart-2]) + " " + str(yArray[i-fileCountStart-2]) + " " + str(zArray0) + " )\n"
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
        
        # for i in range(65,65+xElements+1):
        #     CoorListElement05 = "( " + str(xArray[i-65]) + " " + str(yArray[i-65]-0.0004) + " " + str(zArray05) + " )\n"
        #     sampleFileCont.insert(i, CoorListElement05)
    
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
        OFfile = open("OpenFOAMRunHPC.py")
        OFstring = OFfile.readlines()
        OFfile.close()
        OFstring[12] ="    return_code = subprocess.Popen('sbatch ./../caseDatabase/"+NewDirectory+"/Allrun.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()\n"
        OFstring[15] ="    return_code = subprocess.Popen('sbatch ./../caseDatabase/"+NewDirectory+"/Preproc.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()\n"
        OFstring[18] ="    return_code = subprocess.Popen('sbatch ./../caseDatabase/"+NewDirectory+"/runMesh.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()\n"
        OFstring[19] ="    # Move the cellZones file to the corresponding folder\n"
        OFstring[22]= "    shutil.move('/lustrehome/home/s.2115589/caseDatabase/"+NewDirectory+"/cellZones', NewPath + '/constant/polyMesh/cellZones')\n"
        OFfile = open("OpenFOAMRunHPC.py", "w")
        newOFcont = "".join(OFstring)
        OFfile.write(newOFcont)
        OFfile.close()
        
        # Update the runMesh script to run each new case
        with open("../caseDatabase/"+NewDirectory+"/runMesh.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/runMesh.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        # Update the Preproc script to  each new case
        with open("../caseDatabase/"+NewDirectory+"/Preproc.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/Preproc.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        # Update the Postproc script to  each new case
        with open("../caseDatabase/"+NewDirectory+"/Postproc.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/Postproc.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        # Update the Allrun files script to run each new case
        with open("../caseDatabase/"+NewDirectory+"/Allrun.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/Allrun.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        
        # When this script is finished, no data is required
        SkipStep = "no"
        return SkipStep, NewPath
    
# Create the Parametrization function for worklineMainV1 to call
def CopyUpdateMultiAW(xElements, Amp1, WaveL1, Amp2, WaveL2, xArray, zArray0, zArray1, zArray05):
    # Create the yArray
    yArray = np.empty([len(xArray)])
    for i in range(len(xArray)): 
        yArray[i] = (-Amp1*np.cos(xArray[i]*WaveL1*np.pi)) + (-Amp2*np.cos(xArray[i]*WaveL2*np.pi))+Amp1+Amp2+0.002
    # Copy the files from the base case to a new folder with "Amp" and "WaveL" as name
    SourceDir = 'baseHFMFiles'
    # Transform the float values of amplitude and wavelength into string to
    # name the folders
    Amp = str(Amp1).replace('.', '-') + '_' + str(Amp2).replace('.', '-')
    WaveL = str(WaveL1).replace('.', '-') + '_' + str(WaveL2).replace('.', '-')
    NewDirectory = "A"+Amp+"_W"+WaveL
    ParentDir = '/lustrehome/home/s.2115589/caseDatabase/'
    NewPath = os.path.join(ParentDir, NewDirectory)
    #if path already exists, remove it before copying with copytree()
    if os.path.exists(NewPath):
        print("The folder '% s' already exists, no further action needed..." % NewDirectory)
        SkipStep = "yes"
        return SkipStep, NewPath
    if (Amp1 == 0.0 and WaveL1 != 0.0) or (Amp1 != 0.0 and WaveL1 == 0.0):
        print("The case '% s' already exists, no further action needed..." % NewDirectory)
        SkipStep = "yes"
        return SkipStep, NewPath
    # Create a list with both old & new data sampling
    DirList = glob.glob(ParentDir + "*")
    # Call for the amplitude and wavelength data from each and every case in existance
    for Diri in DirList:
        AWfile = np.genfromtxt(Diri+'/AmpWaveL.txt', delimiter=' ')
        if (Amp2 == 0.0 or WaveL2 == 0.0) and (AWfile[0,0] == Amp1 and AWfile[1,0] == WaveL1):
            print("The case '% s' already exists, no further action needed..." % NewDirectory)
            SkipStep = "yes"
            return SkipStep, NewPath
    else:
        shutil.copytree(SourceDir, NewPath)
        print("Simulation files successfully copied to directory '% s'..." % NewDirectory)
        
        # Write a file with the Amp and WaveL data for future reference within the NN
        AWfile = open("../caseDatabase/"+NewDirectory+"/AmpWaveL.txt", "w")
        AWfile.write(str(Amp1)+" " + str(Amp2) +"\n")
        AWfile.write(str(WaveL1)+" "+str(WaveL2)+"\n")
        AWfile.close()
        AWfile = open("../caseDatabase/"+NewDirectory+"/AmpWaveLNN.txt", "w")
        AWfile.write(str(Amp1)+" " + str(Amp2) +" ")
        AWfile.write(str(WaveL1)+" "+str(WaveL2)+"\n")
        AWfile.close()
        
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
        fileCountStart = 56
        meshFileCont.insert(fileCountStart, 'polyLine 3 2\n')
        meshFileCont.insert(fileCountStart+1, '(\n')
        for i in range(fileCountStart+2,fileCountStart+xElements+3):
            CoorListElement0 = "( " + str(xArray[i-fileCountStart-2]) + " " + str(yArray[i-fileCountStart-2]) + " " + str(zArray0) + " )\n"
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
        
        # for i in range(65,65+xElements+1):
        #     CoorListElement05 = "( " + str(xArray[i-65]) + " " + str(yArray[i-65]-0.0004) + " " + str(zArray05) + " )\n"
        #     sampleFileCont.insert(i, CoorListElement05)
    
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
        OFfile = open("OpenFOAMRunHPC.py")
        OFstring = OFfile.readlines()
        OFfile.close()
        OFstring[12] ="    return_code = subprocess.Popen('sbatch ./../caseDatabase/"+NewDirectory+"/Allrun.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()\n"
        OFstring[15] ="    return_code = subprocess.Popen('sbatch ./../caseDatabase/"+NewDirectory+"/Preproc.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()\n"
        OFstring[18] ="    subprocess.Popen(['sbatch', './../caseDatabase/"+NewDirectory+"/runMesh.sh'])\n"
        OFstring[19] ="    # Move the cellZones file to the corresponding folder\n"
        OFstring[22]= "    shutil.move('/lustrehome/home/s.2115589/caseDatabase/"+NewDirectory+"/cellZones', NewPath + '/constant/polyMesh/cellZones')\n"
        OFfile = open("OpenFOAMRunHPC.py", "w")
        newOFcont = "".join(OFstring)
        OFfile.write(newOFcont)
        OFfile.close()
        
        # Update the runMesh script to run each new case
        with open("../caseDatabase/"+NewDirectory+"/runMesh.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/runMesh.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        # Update the Preproc script to  each new case
        with open("../caseDatabase/"+NewDirectory+"/Preproc.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/Preproc.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        # Update the Postproc script to  each new case
        with open("../caseDatabase/"+NewDirectory+"/Postproc.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/Postproc.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        # Update the Allrun files script to run each new case
        with open("../caseDatabase/"+NewDirectory+"/Allrun.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/Allrun.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        
        # When this script is finished, no data is required
        SkipStep = "no"
        return SkipStep, NewPath

# Create the Parametrization function for worklineMainV1 to call
def CopyUpdateNerves(xElements, L, xCn, yCn, xArray, zArray0, zArray1, zArray05):
    # Create the SpLine
    Dx = (xElements/L)*max(xCn)
    f = CubicSpline(xCn, yCn)
    xSpLine = np.linspace(min(xCn), max(xCn), Dx)
    ySpLine = f(xSpLine)
    # Create the yArray
    yCount = xElements/(Dx*2)
    ySpLine.append(np.flip(ySpLine))
    for count in range(yCount):
        yArray = ySpLine + ySpLine
    # Copy the files from the base case to a new folder with "Amp" and "WaveL" as name
    SourceDir = 'baseHFMFiles'
    # Transform the float values of amplitude and wavelength into string to
    # name the folders
    Amp = str(xCn).replace('.', '-')
    WaveL = str(yCn).replace('.', '-')
    NewDirectory = "x"+Amp+"_y"+WaveL
    ParentDir = '/lustrehome/home/s.2115589/caseDatabase/'
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
        fileCountStart = 56
        meshFileCont.insert(fileCountStart, 'polyLine 3 2\n')
        meshFileCont.insert(fileCountStart+1, '(\n')
        for i in range(fileCountStart+2,fileCountStart+xElements+3):
            CoorListElement0 = "( " + str(xArray[i-fileCountStart-2]) + " " + str(yArray[i-fileCountStart-2]) + " " + str(zArray0) + " )\n"
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
        
        # for i in range(65,65+xElements+1):
        #     CoorListElement05 = "( " + str(xArray[i-65]) + " " + str(yArray[i-65]-0.0004) + " " + str(zArray05) + " )\n"
        #     sampleFileCont.insert(i, CoorListElement05)
    
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
        OFfile = open("OpenFOAMRunHPC.py")
        OFstring = OFfile.readlines()
        OFfile.close()
        OFstring[12] ="    return_code = subprocess.Popen('sbatch ./../caseDatabase/"+NewDirectory+"/Allrun.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()\n"
        OFstring[15] ="    return_code = subprocess.Popen('sbatch ./../caseDatabase/"+NewDirectory+"/Preproc.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()\n"
        OFstring[18] ="    return_code = subprocess.Popen('sbatch ./../caseDatabase/"+NewDirectory+"/runMesh.sh'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()\n"
        OFstring[19] ="    # Move the cellZones file to the corresponding folder\n"
        OFstring[22]= "    shutil.move('/lustrehome/home/s.2115589/caseDatabase/"+NewDirectory+"/cellZones', NewPath + '/constant/polyMesh/cellZones')\n"
        OFfile = open("OpenFOAMRunHPC.py", "w")
        newOFcont = "".join(OFstring)
        OFfile.write(newOFcont)
        OFfile.close()
        
        # Update the runMesh script to run each new case
        with open("../caseDatabase/"+NewDirectory+"/runMesh.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/runMesh.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        # Update the Preproc script to  each new case
        with open("../caseDatabase/"+NewDirectory+"/Preproc.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/Preproc.sh'.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        # Update the Postproc script to  each new case
        with open("../caseDatabase/"+NewDirectory+"/Postproc.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/Postproc.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        # Update the Allrun files script to run each new case
        with open("../caseDatabase/"+NewDirectory+"/Allrun.sh", "rt") as bat_file:
            text = bat_file.readlines()

        new_text = []
        for line in text:
            if "cd /lustrehome/home/s.2115589/caseDatabase/" in line:
                new_text.append(line.replace("cd /lustrehome/home/s.2115589/caseDatabase/", "cd /lustrehome/home/s.2115589/caseDatabase/"+NewDirectory))
            else:
                new_text.append(line)

        with open("../caseDatabase/"+NewDirectory+"/Allrun.sh", "wt") as bat_file:
            for line in new_text:
                bat_file.write(line)
        
        # When this script is finished, no data is required
        SkipStep = "no"
        return SkipStep, NewPath