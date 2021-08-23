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

trueScale=np.array([9369,7487,9369,7487])
dispScale=np.array([255,255,255,255])
#iterate through flourescent-nissl pairs
for imPathF,imPathN in zip(flour,nissl):
    #print(imPathF,imPathN)

    #load and process flourescent image for network input
    imF = tf.keras.preprocessing.image.load_img(imPathF,target_size=targetShape)
    imF = tf.keras.preprocessing.image.img_to_array(imF)
    imF = np.array(imF)
    imF=np.expand_dims(imF,0)
    imF=np.array(imF,dtype='float32')/255.00
    
    #load and process nissl image for network input
    imN = tf.keras.preprocessing.image.load_img(imPathN,target_size=targetShape)
    imN = tf.keras.preprocessing.image.img_to_array(imN)
    imN = np.array(imN)
    imN=np.expand_dims(imN,0)
    imN=np.array(imN,dtype='float32')/255.00
    
    #get LC boundary predictions from network, assign to coordinate variables
    fPred=cnn.predict(imF)[0]
    nPred=cnn.predict(imN)[0]
  
     #clear variables holding the images
    del imN
    del imF
    
  
    #load and process flourescent image for network input
    imF = tf.keras.preprocessing.image.load_img(imPathF)
    imF = tf.keras.preprocessing.image.img_to_array(imF)
    imF = np.array(imF)
    imF=np.array(imF,dtype='float32')/255.00
    
    #load and process nissl image for network input
    imN = tf.keras.preprocessing.image.load_img(imPathN)
    imN = tf.keras.preprocessing.image.img_to_array(imN)
    imN = np.array(imN)
    imN=np.array(imN,dtype='float32')/255.00
  
    
  
    
    # imN=imN[0,:,:,:]
    # imF=imF[0,:,:,:]
       
    fTargs=annots.loc[imPathF==annots.Name,:]
    nTargs=annots.loc[imPathN==annots.Name,:]
    
    
    fLabels=int(fTargs.x1),int(fTargs.y1),int(fTargs.x2),int(fTargs.y2)
    nLabels=int(nTargs.x1),int(nTargs.y1),int(nTargs.x2),int(nTargs.y2)
    
    
    fLabels=np.divide(fLabels,trueScale)
    nLabels=np.divide(nLabels,trueScale)   
    
    fCompA=np.multiply(fLabels,trueScale).astype(int)
    nCompA=np.multiply(nLabels,trueScale).astype(int)
    fCompP=np.multiply(fPred,trueScale).astype(int)
    nCompP=np.multiply(nPred,trueScale).astype(int)
    

    #Euc distance px errors
    d1=sqrt(((fCompA[0]-fCompP[0])**2)+((fCompA[1]-fCompP[1])**2))
    d2=sqrt(((fCompA[2]-fCompP[2])**2)+((fCompA[3]-fCompP[3])**2))
    
    #print(d1,d2)
    print("Flour: "+ str(mean([d1,d2])))
    
    d1=sqrt(((nCompA[0]-nCompP[0])**2)+((nCompA[1]-nCompP[1])**2))
    d2=sqrt(((nCompA[2]-nCompP[2])**2)+((nCompA[3]-nCompP[3])**2))
    
    #print(d1,d2)
    print("Nissil: " + str(mean([d1,d2])))
  
    
  
    fShowA=np.multiply(fLabels,dispScale).astype(int)
    nShowA=np.multiply(nLabels,dispScale).astype(int)
    fShowP=np.multiply(fPred,dispScale).astype(int)
    nShowP=np.multiply(nPred,dispScale).astype(int)
    
    
    # plt.pyplot.figure()
    # cv2.rectangle(imF,tuple(fShowA[0:2]),tuple(fShowA[2:4]),(0,0,255),3)
    # cv2.rectangle(imF,tuple(fShowP[0:2]),tuple(fShowP[2:4]),(255,0,0),2)
    # plt.pyplot.imshow(imF)
    
    # plt.pyplot.figure()
    # cv2.rectangle(imN,tuple(nShowA[0:2]),tuple(nShowA[2:4]),(0,0,255),3)
    # cv2.rectangle(imN,tuple(nShowP[0:2]),tuple(nShowP[2:4]),(255,0,0),2)  
    # plt.pyplot.imshow(imN)
    
    plt.pyplot.figure()
    cv2.rectangle(imF,tuple(fCompA[0:2]),tuple(fCompA[2:4]),(0,0,255),60)
    cv2.rectangle(imF,tuple(fCompP[0:2]),tuple(fCompP[2:4]),(255,0,0),40)
    plt.pyplot.imshow(imF)
    
    plt.pyplot.figure()
    cv2.rectangle(imN,tuple(nCompA[0:2]),tuple(nCompA[2:4]),(0,0,255),60)
    cv2.rectangle(imN,tuple(nCompP[0:2]),tuple(nCompP[2:4]),(255,0,0),40)  
    plt.pyplot.imshow(imN)
    
    
    # #clear variables holding the images
    # del imN
    # del imF
  













