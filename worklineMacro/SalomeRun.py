"""
2. SalomeRun.py Opens Salome to run the SalomeGeom.py script and build the
    geometry used for topology application on the Helium and Solid regions.
@author: Daniela Segura
"""
# Subprocess and time libraries to run terminal commands and software
import subprocess
import shutil

def RunSalome(Amp, WaveL, xEl, NewPath):
    print("Running Salome to create the geometry for the Helium topology...")
    # Update the SalomeGeom.py to build the curve with the Amplitude and Wavelength data
    AWfile = open("SalomeGeom.py")
    AWstring = AWfile.readlines()
    AWfile.close()
    AWstring[34] ="Curve_wall = geompy.MakeCurveParametric('t', '-"+str(Amp)+"*cos(t*"+str(WaveL)+"*pi)+0.2+"+str(Amp)+"-0.00013','0.05',0.0,2.0,"+str(xEl)+",GEOM.Polyline,theNewMethod = True)\n"
    AWfile = open("SalomeGeom.py", "w")
    newAWcont = "".join(AWstring)
    AWfile.write(newAWcont)
    AWfile.close()
    # Make a subprocess call to execute salome with the SalomeGeom.py script
    subprocess.call(['./../../SALOME-9.7.0-native-UB20.04-SRC/salome -t python SalomeGeom.py'], shell=True)
    shutil.move("/media/dani/Data/Ubuntufiles/ProperThermoProp/worklineMacro/HeliumRegion.stl", NewPath + "/HeliumRegion.stl")

    print("Geometry succesfully created and moved to the new directory...")
