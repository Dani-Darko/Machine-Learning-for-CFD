#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:05:44 2022

@author: dani
"""
import glob
import os
# Create a list with both old & new data sampling
DirList1 = glob.glob("../caseDatabase/*")
DirList = []
for Dir1 in DirList1:
    if os.path.exists(Dir1 + '/log.chtMultiRegionSimpleFoam') and not os.path.exists(Dir1 + '/5500'):
        DirList.append(Dir1)
print(DirList)