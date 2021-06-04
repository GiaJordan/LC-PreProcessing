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
import pandas as pd
from shutil import copyfile

targetShape=(224,224)
    
basePath=r"C:\Users\giajordan\Documents\GitHub\LC-PreProcessing"
imPath=os.path.sep.join([basePath,r'Images'])
annotPath=os.path.sep.join([basePath,r'Data.csv'])
    
    
outputBase=r'Output'
modelPath=os.path.sep.join([basePath,outputBase,'locator.h5'])
plotPath=os.path.sep.join([basePath,outputBase,'plot.png'])
testNames=os.path.sep.join([basePath,outputBase,'testImages.txt'])



cnn=tf.keras.models.load_model(modelPath)

os.chdir(r'C:\Gianna\MMU_35765')

files=os.listdir()

for file in files:
    
    if os.path.isfile(file) & file.endswith('.xlsx') & file.startswith('MMU'):
        csv=pd.read_excel(file)
        coords=np.array(csv.iloc[:,1])
        diff=np.diff(coords,axis=-1)
        iF=np.append(diff,0)
        iN=np.insert(diff,0,0)

        flour=csv.iloc[iF==30,-1].values.tolist()
        nissl=csv.iloc[iN==30,-1].values.tolist()
    elif os.path.isdir(file):
        imagesPath=file

# imPathB=r"C:\\Users\\giajordan\\Documents\\GitHub\\LC-PreProcessing\\Images\\b26166_Z00.ome.tif"
# imPathN=r"C:\\Users\\giajordan\\Documents\\GitHub\\LC-PreProcessing\\Images\\b26166_Z01.ome.tif"

os.chdir(imagesPath)

if 'ScaledImages' not in os.listdir():
    os.mkdir('ScaledImages')
    originalImages=csv.iloc[iN!=30,-1].values.tolist()
    
    for image in originalImages:
        copyfile(image,os.path.join('ScaledImages',image))
        

flour.pop(2)
nissl.pop(2)
for imPathB,imPathN in zip(flour,nissl):
    #print(imPathB,imPathN)
    #imB = cv2.imread(imPathB)
    imB = tf.keras.preprocessing.image.load_img(imPathB,target_size=targetShape)
    imB = tf.keras.preprocessing.image.img_to_array(imB)
    imB = np.array(imB)
    imB=np.expand_dims(imB,0)
    imB=np.array(imB,dtype='float32')/255.00
    
    #imN = cv2.imread(imPathN)
    imN = tf.keras.preprocessing.image.load_img(imPathN,target_size=targetShape)
    imN = tf.keras.preprocessing.image.img_to_array(imN)
    imN = np.array(imN)
    imN=np.expand_dims(imN,0)
    imN=np.array(imN,dtype='float32')/255.00
    
    [[bx1,by1,bx2,by2]]=cnn.predict(imB)
    [[nx1,ny1,nx2,ny2]]=cnn.predict(imN)
    
    
    (bh,bw) = imB.shape[:2]
    (nh,nw) = imN.shape[:2]
    
    del imN
    del imB
  
    bxDiff=(bx2-bx1)
    byDiff=(by1-by2)
    
    nxDiff=(nx2-nx1)
    nyDiff=(ny1-ny2)
    
    
    
    xScaleFac=(bxDiff/nxDiff)
    yScaleFac=(byDiff/nyDiff)
    # xScaleFac=(nxDiff/bxDiff)
    # yScaleFac=(nyDiff/byDiff)
    
    nisslIm=pil.Image.open(imPathN)
    w,h=nisslIm.width,nisslIm.height
    nisslIm=nisslIm.resize((floor(w*xScaleFac),floor(h*yScaleFac)),resample=pil.Image.LANCZOS)
    #nisslIm.resize((floor(w*xScaleFac),floor(h*yScaleFac)),resample=pil.Image.LANCZOS).show()
    nisslIm.save(os.path.join('ScaledImages',imPathN))
    del bx1,by1,bx2,by2,nx1,ny1,nx2,ny2











