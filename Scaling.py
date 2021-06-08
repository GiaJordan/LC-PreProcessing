# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:04:49 2021

@author: giajordan
to be ran in data directory with excel spreadsheet and images subdirectory
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

#path to github repo
basePath=r"C:\Users\giajordan\Documents\GitHub\LC-PreProcessing"


#pathing
imPath=os.path.sep.join([basePath,r'Images'])
annotPath=os.path.sep.join([basePath,r'Data.csv'])
outputBase=r'Output'
modelPath=os.path.sep.join([basePath,outputBase,'locator.h5'])
plotPath=os.path.sep.join([basePath,outputBase,'plot.png'])
testNames=os.path.sep.join([basePath,outputBase,'testImages.txt'])

targetShape=(224,224)

#load model
cnn=tf.keras.models.load_model(modelPath)

#os.chdir(r'C:\Gianna\MMU_35765')

#get list of directory contents and iterate through
files=os.listdir()
for file in files:
    
    #If current file is excel spreadsheet, open and identify flourescent-nissl image pairs and save to variables
    if os.path.isfile(file) & file.endswith('.xlsx') & file.startswith('MMU'):
        csv=pd.read_excel(file)
        coords=np.array(csv.iloc[:,1])
        diff=np.diff(coords,axis=-1)
        iF=np.append(diff,0)
        iN=np.insert(diff,0,0)

        flour=csv.iloc[iF==30,-1].values.tolist()
        nissl=csv.iloc[iN==30,-1].values.tolist()
    #if 'file' is actually a subdirectory, set as path to image folder
    elif os.path.isdir(file):
        imagesPath=file


#move to images subdirectory
os.chdir(imagesPath)

#If first time running, create subdirectory for scaled images and copy original images to subdirectory
if 'ScaledImages' not in os.listdir():
    os.mkdir('ScaledImages')
    originalImages=csv.iloc[iN!=30,-1].values.tolist()
    
    for image in originalImages:
        copyfile(image,os.path.join('ScaledImages',image))
        
# this is just for the test directory I used to build the script, was missing image names
# flour.pop(2)
# nissl.pop(2)

#iterate through flourescent-nissl pairs
for imPathB,imPathN in zip(flour,nissl):
    #print(imPathB,imPathN)

    #load and process flourescent image for network input
    imB = tf.keras.preprocessing.image.load_img(imPathB,target_size=targetShape)
    imB = tf.keras.preprocessing.image.img_to_array(imB)
    imB = np.array(imB)
    imB=np.expand_dims(imB,0)
    imB=np.array(imB,dtype='float32')/255.00
    
    #load and process nissl image for network input
    imN = tf.keras.preprocessing.image.load_img(imPathN,target_size=targetShape)
    imN = tf.keras.preprocessing.image.img_to_array(imN)
    imN = np.array(imN)
    imN=np.expand_dims(imN,0)
    imN=np.array(imN,dtype='float32')/255.00
    
    #get LC boundary predictions from network, assign to coordinate variables
    [[bx1,by1,bx2,by2]]=cnn.predict(imB)
    [[nx1,ny1,nx2,ny2]]=cnn.predict(imN)
    
    #get height, width of images
    (bh,bw) = imB.shape[:2]
    (nh,nw) = imN.shape[:2]
    
    #clear variables holding the images
    del imN
    del imB
  
    #calculate widths and heights of LC boundary for each image
    bxDiff=(bx2-bx1)
    byDiff=(by1-by2)
    nxDiff=(nx2-nx1)
    nyDiff=(ny1-ny2)
    
    
    #get size ratio of boundarys for each image
    xScaleFac=(bxDiff/nxDiff)
    yScaleFac=(byDiff/nyDiff)
    # xScaleFac=(nxDiff/bxDiff)
    # yScaleFac=(nyDiff/byDiff)
    
    #open full size nissl image
    nisslIm=pil.Image.open(imPathN)
    
    #get width and height
    w,h=nisslIm.width,nisslIm.height
    
    #resize nissl image
    nisslIm=nisslIm.resize((floor(w*xScaleFac),floor(h*yScaleFac)),resample=pil.Image.LANCZOS)
    
    ##alternatively, display resized nissl image
    #nisslIm.resize((floor(w*xScaleFac),floor(h*yScaleFac)),resample=pil.Image.LANCZOS).show()
    
    #save in scaled images directory
    nisslIm.save(os.path.join('ScaledImages',imPathN))
    
    #clear vars
    del bx1,by1,bx2,by2,nx1,ny1,nx2,ny2











