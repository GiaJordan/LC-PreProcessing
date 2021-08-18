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
from math import floor, sqrt
from statistics import median, mean
#path to github repo
basePath=r"C:\Users\giajordan\Documents\GitHub\LC-PreProcessing"


#pathing
imPath=os.path.sep.join([basePath,r'Images'])
#annotPath=os.path.sep.join([basePath,r'Data.csv'])
outputBase=r'Output'
modelPath=os.path.sep.join([basePath,outputBase,'locatorOPT.h5'])
plotPath=os.path.sep.join([basePath,outputBase,'plot.png'])
testNames=os.path.sep.join([basePath,outputBase,'testImages.txt'])

targetShape=(224,224)

#load model
cnn=tf.keras.models.load_model(modelPath)

os.chdir(r'C:\Gianna\MMU_35765')



annotPath=os.path.sep.join([os.getcwd(),r'Data.csv'])

annots=pd.read_csv(annotPath)


#get list of directory contents and iterate through
files=os.listdir()
for file in files:
    
    #If current file is excel spreadsheet, open and identify flourescent-nissl image pairs and save to variables
    if os.path.isfile(file) & file.endswith('.xlsx') & file.startswith('MMU'):
        csv=pd.read_excel(file)
        coords=np.array(csv.iloc[:,1])
        diff=np.diff(coords,axis=-1)
        iN=np.append(diff,0)
        iF=np.insert(diff,0,0)

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
        if not np.isnan(image):
            copyfile(image,os.path.join('ScaledImages',image))
        
# this is just for the test directory I used to build the script, was missing image names
flour.pop(2)
nissl.pop(2)

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
    bPred=cnn.predict(imB)
    nPred=cnn.predict(imN)
 
    [[bx1,by1,bx2,by2]]=bPred
    [[nx1,ny1,nx2,ny2]]=nPred
 
    
    #get height, width of images
    (bh,bw) = imB.shape[:2]
    (nh,nw) = imN.shape[:2]
    
    imN=imN[0,:,:,:]
    imB=imB[0,:,:,:]
       
    bTargs=annots.loc[imPathB==annots.Name,:]
    nTargs=annots.loc[imPathN==annots.Name,:]
    
    # bt1,bt2,bt3,bt4=int(bTargs.x1),int(bTargs.y1),int(bTargs.x2),int(bTargs.y2)
    # nt1,nt2,nt3,nt4=int(nTargs.x1),int(nTargs.y1),int(nTargs.x2),int(nTargs.y2)
    
    # bLabels=[bt1,bt2,bt3,bt4]
    # nLabels=[nt1,nt2,nt3,nt4]
    
    # print(bLabels,bPred,nLabels,nPred)
    
    # d1=sqrt(((bx1-bt1)**2)+((by1-bt2)**2))
    # d2=sqrt(((bx2-bt3)**2)+((by2-bt4)**2))
    
    # print(d1,d2)
    # print("Flour: "+ str(mean([d1,d2])))
    
    # d1=sqrt(((nx1-nt1)**2)+((ny1-nt2)**2))
    # d2=sqrt(((nx2-nt3)**2)+((ny2-nt4)**2))
    
    # print(d1,d2)
    # print("Nissil: " + str(mean([d1,d2])))
  
    
    bx1,by1,bx2,by2=int(bx1*(225)),int(by1*(225)),int(bx2*(225)),int(by2*(225))    
    nx1,ny1,nx2,ny2=int(nx1*(225)),int(ny1*(225)),int(nx2*(225)),int(ny2*(225))  
    
    
    bt1,bt2,bt3,bt4=int(bTargs.x1*(255/9369)),int(bTargs.y1*(255/7487)),int(bTargs.x2*(255/9369)),int(bTargs.y2*(255/7487))
    nt1,nt2,nt3,nt4=int(nTargs.x1*(255/9369)),int(nTargs.y1*(255/7487)),int(nTargs.x2*(255/9369)),int(nTargs.y2*(255/7487))
    
    
    
    plt.pyplot.figure()
    cv2.rectangle(imB,(bt1,bt2),(bt3,bt4),(0,0,255),3)
    cv2.rectangle(imB,(bx1,by1),(bx2,by2),(255,0,0),2)
    plt.pyplot.imshow(imB)
    
    plt.pyplot.figure()
    cv2.rectangle(imN,(nt1,nt2),(nt3,nt4),(0,0,255),3)
    cv2.rectangle(imN,(nx1,ny1),(nx2,ny2),(255,0,0),2)  
    plt.pyplot.imshow(imN)
    
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











