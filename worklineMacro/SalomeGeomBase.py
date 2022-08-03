"""
2. SalomeGeom.py Builts the geometry and exports the STL file for the
    Helium topology application on the Helium and Solid regions within
    the OpenFOAM simulations.

@author: Daniela Segura
"""
# System, path and runtime information
import os
import sys

# salome libraries and GEOM-Python interface
import salome
import GEOM
from salome.geom import geomBuilder

# Call for amplitude and wavelength values from "AWvalues.txt" file
AWfile = open("AWvalues.txt")
AWstring = AWfile.readlines()
AWfile.close()
Amp = float(AWstring[0])
WaveL = float(AWstring[1])

# Initialize salome and the geometry study module
salome.salome_init()
geompy = geomBuilder.New(salome.myStudy)

# Built the vertices, lines and curves
Vertex1 = geompy.MakeVertex(0.0, 0.0, 0.05)
Vertex2 = geompy.MakeVertex(1.0, 0.0, 0.05)
Vertex3 = geompy.MakeVertex(0.0, 0.2, 0.05)
Vertex4 = geompy.MakeVertex(1.0, 0.2, 0.05)
Line_sym = geompy.MakeLineTwoPnt(Vertex1, Vertex2)
Line_inlet = geompy.MakeLineTwoPnt(Vertex1, Vertex3)
Line_outlet = geompy.MakeLineTwoPnt(Vertex2, Vertex4)
Curve_wall = geompy.MakeCurveParametric(
                                        "t",
                                        "-Amp*cos(t*WaveL*pi)+0.2+Amp-0.0001",
                                        "0.05",
                                        0.0,
                                        1.0,
                                        200,
                                        GEOM.Polyline,
                                        theNewMethod = True
                                        )
Contours = [Line_sym, Line_inlet, Curve_wall, Line_outlet]
SurfaceTopoSet = geompy.MakeFilling(Contours)

# Export an STL file into the new folder
geompy.ExportSTL(SurfaceTopoSet, "HeliumRegion.stl")
print("HeliumRegion.stl file properly created in the Macro folder...")






"""
2. SalomeGeom.py Builts the geometry and exports the STL file for the
    Helium topology application on the Helium and Solid regions within
    the OpenFOAM simulations.

@author: Daniela Segura
"""
# System, path and runtime information
import os
import sys

# salome libraries and GEOM-Python interface
import salome
import GEOM
from salome.geom import geomBuilder

# Initialize salome and the geometry study module
salome.salome_init()
geompy = geomBuilder.New(salome.myStudy)

# Built the vertices, lines and curves
Vertex1 = geompy.MakeVertex(0.0, 0.0, 0.05)
Vertex2 = geompy.MakeVertex(1.0, 0.0, 0.05)
Vertex3 = geompy.MakeVertex(0.0, 0.2, 0.05)
Vertex4 = geompy.MakeVertex(1.0, 0.2, 0.05)
Line_sym = geompy.MakeLineTwoPnt(Vertex1, Vertex2)
Line_inlet = geompy.MakeLineTwoPnt(Vertex1, Vertex3)
Line_outlet = geompy.MakeLineTwoPnt(Vertex2, Vertex4)
Curve_wall = geompy.MakeCurveParametric("t", "-Amp*cos(t*WaveL*pi)+0.2+Amp-0.0001","0.05",0.0,1.0,200,GEOM.Polyline,theNewMethod = True)
Contours = [Line_sym, Line_inlet, Curve_wall, Line_outlet]
SurfaceTopoSet = geompy.MakeFilling(Contours)

# Export an STL file into the new folder
geompy.ExportSTL(SurfaceTopoSet, "HeliumRegion.stl")
print("HeliumRegion.stl file properly created in the Macro folder...")