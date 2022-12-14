"""
2. SalomeGeom.py Builts the geometry and exports the STL file for the
    Helium topology application on the Helium and Solid regions within
    the OpenFOAM simulations.

@author: Daniela Segura
"""

# salome libraries and GEOM-Python interface
import salome
import GEOM
from salome.geom import geomBuilder

# Initialize salome and the geometry study module
salome.salome_init()
geompy = geomBuilder.New(salome.myStudy)

mindeg = -360
maxdeg = 360
tol3d  = 0.000000001
tol2d  = 0.000000001
nbiter = 0

# Built the vertices, lines and curves
Vertex1 = geompy.MakeVertex(-0.5, 0.0, 0.05)
Vertex2 = geompy.MakeVertex(2.5, 0.0, 0.05)
Vertex3 = geompy.MakeVertex(-0.5, 0.199995, 0.05)
Vertex4 = geompy.MakeVertex(2.5, 0.199995, 0.05)
Vertex5 = geompy.MakeVertex(0.0, 0.199995, 0.05)
Vertex6 = geompy.MakeVertex(2.0, 0.199995, 0.05)
Line_sym = geompy.MakeLineTwoPnt(Vertex1, Vertex2)
Line_inExt = geompy.MakeLineTwoPnt(Vertex3, Vertex5)
Line_outExt = geompy.MakeLineTwoPnt(Vertex6, Vertex4)
Curve_wall = geompy.MakeCurveParametric('t', '-0.001*cos(t*16.0*pi)+0.2+0.001-0.000005','0.05',0.0,2.0,600,GEOM.Polyline,theNewMethod = True)
Wall = geompy.MakeWire([Line_inExt, Curve_wall, Line_outExt])
Contours = [Line_sym, Wall]
compound = geompy.MakeCompound(Contours)
SurfaceTopoSet = geompy.MakeFilling(compound, mindeg, maxdeg, tol3d, tol2d, nbiter, theMethod=GEOM.FOM_AutoCorrect)

# Export an STL file into the new folder
geompy.ExportSTL(SurfaceTopoSet, "HeliumRegion.stl", theIsASCII=True, theDeflection=0.000000001)
print("HeliumRegion.stl file properly created in the Macro folder...")