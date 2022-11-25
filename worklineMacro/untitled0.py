#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:20:23 2022

@author: dani
"""

import glob
import os
# Create a list with both old & new data sampling
DirList1 = glob.glob("../caseDatabase/*")
DirList = []
NoList = []
for Dir1 in DirList1:
    # Make a subprocess call to the openfoam2106 terminal and run the relevant ./Allrun
    DirList1 = glob.glob("../caseDatabase/*")
    if os.path.exists(Dir1 + '/1000/Helium/U'):
        DirList.append(Dir1)
        OFfile = open(Dir1 +"/1000/Helium/U")
        OFstring = OFfile.readlines()
        for i in OFstring:
            # print(i)
            if i == "internalField   uniform (0 0 0);":
                NoList.append(Dir1)
                break
print("The amount of available cases is '% s'" % len(DirList))
# print(DirList)
print("The amount of failed sims is '% s'" % len(NoList))
# print(DirList)