# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:04:49 2021

@author: giajordan
"""



import os
import tensorflow as tf
import matplotlib as plt
import cv2
import numpy as np
import PIL as pil
from math import floor

targetShape=(224,224)
    
basePath=r"C:\Users\giajordan\Documents\GitHub\LC-PreProcessing"
imPath=os.path.sep.join([basePath,r'Images'])
annotPath=os.path.sep.join([basePath,r'Data.csv'])
    
    
outputBase=r'Output'
modelPath=os.path.sep.join([basePath,outputBase,'locator.h5'])
plotPath=os.path.sep.join([basePath,outputBase,'plot.png'])
testNames=os.path.sep.join([basePath,outputBase,'testImages.txt'])



cnn=tf.keras.models.load_model(modelPath)


imPathB=r"C:\\Users\\giajordan\\Documents\\GitHub\\LC-PreProcessing\\Images\\b26166_Z00.ome.tif"
imPathN=r"C:\\Users\\giajordan\\Documents\\GitHub\\LC-PreProcessing\\Images\\b26166_Z01.ome.tif"


#imB = cv2.imread(imPathB)
imB = tf.keras.preprocessing.image.load_img(imPathB,target_size=targetShape)
imB = tf.keras.preprocessing.image.img_to_array(imB)
imB = np.array(imB)
imB=np.expand_dims(imB,axis=0)

#imN = cv2.imread(imPathN)
imN = tf.keras.preprocessing.image.load_img(imPathN,target_size=targetShape)
imN = tf.keras.preprocessing.image.img_to_array(imN)
imN = np.array(imN)
imN=np.expand_dims(imN,axis=0)

[[bx1,by1,bx2,by2]]=cnn.predict(imB)
[[nx1,ny1,nx2,ny2]]=cnn.predict(imN)


(bh,bw) = imB.shape[:2]
(nh,nw) = imN.shape[:2]

del imN
del imB

bxDiff=(bx2-bx1)*bw
byDiff=(by2-by1)*bh

nxDiff=(nx2-nx1)*nw
nyDiff=(ny2-ny1)*nh



xScaleFac=(bxDiff/nxDiff)
yScaleFac=(byDiff/nyDiff)

nisslIm=pil.Image.open(imPathN)
w,h=nisslIm.width,nisslIm.height
#nisslIm=nisslIm.resize(floor(w*xScaleFac),floor(h*yScaleFac))
nisslIm.resize(floor(w*xScaleFac),floor(h*yScaleFac)).show()












