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

mindeg = 2
maxdeg = 5
tol3d  = 0.0001
tol2d  = 0.0001
nbiter = 0

# Built the vertices, lines and curves
Vertex1 = geompy.MakeVertex(0.0, 0.0, 0.05)
Vertex2 = geompy.MakeVertex(2.0, 0.0, 0.05)
Vertex3 = geompy.MakeVertex(0.0, 0.2, 0.05)
Vertex4 = geompy.MakeVertex(2.0, 0.2, 0.05)
Line_sym = geompy.MakeLineTwoPnt(Vertex1, Vertex2)
Line_inlet = geompy.MakeLineTwoPnt(Vertex1, Vertex3)
Line_outlet = geompy.MakeLineTwoPnt(Vertex2, Vertex4)
Curve_wall = geompy.MakeCurveParametric('t', '-0.0019*cos(t*22.0*pi)+0.2+0.0019-0.00013','0.05',0.0,2.0,300,GEOM.Polyline,theNewMethod = True)
Contours = [Line_sym, Curve_wall]
compound = geompy.MakeCompound(Contours)
SurfaceTopoSet = geompy.MakeFilling(compound, mindeg, maxdeg, tol3d, tol2d, nbiter, theMethod=GEOM.FOM_AutoCorrect)

# Export an STL file into the new folder
geompy.ExportSTL(SurfaceTopoSet, "HeliumRegion.stl")
print("HeliumRegion.stl file properly created in the Macro folder...")