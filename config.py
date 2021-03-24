# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os




basePath=r"C:\Users\giajordan\Documents\GitHub\LC-PreProcessing"
imPath=os.path.sep.join([basePath,r'Images'])
annotPath=os.path.sep.join([basePath,r'Data.csv'])


outputBase=r'Output'
modelPtah=os.path.sep.join([basePath,outputBase,'locator.h5'])
plotPath=os.path.sep.join([basePath,outputBase,'plot.png'])
testNames=os.path.sep.join([basePath,outputBase,'testImages.txt'])


initLR=1e-4
epochs=5
batchSize=1





