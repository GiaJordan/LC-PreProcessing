# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:35:42 2021

@author: giajordan
"""


import os
import tensorflow as tf
import matplotlib as plt
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from math import floor
import PIL as pil
from statistics import median
import matplotlib.pyplot as plt


physDevs=tf.config.list_physical_devices('GPU')

    
   



if __name__ == "__main__":
    
    #Training Parameters
    noEpochs=1030000
    targetShape=(224,224)
    lR=1.0#0.005
    tolerance=15
    batchSize=25
    delta=.0005
    
    #Location of repo, probably dont need later
    basePath=r"C:\Users\giajordan\Documents\GitHub\LC-PreProcessing"
    
    noUnits=128
    noLayers=1
    #acFunc='sigmoid'
    acFunc='sigmoid'
    #optimize=tf.keras.optimizers.Adam(learning_rate=lR)
    optimize=tf.keras.optimizers.Adagrad(learning_rate=lR)
    #lossfxn='mse'
    lossfxn='MeanAbsolutePercentageError'
    
    #filtNo=3
    #kSize=(35,35)
    
    
    unitsPerLayer=int(noUnits/noLayers)
    
    
    
    #pathing
    imPath=os.path.sep.join([basePath,r'Images'])
    annotPath=os.path.sep.join([basePath,r'Data.csv'])
    
    outputBase=r'Output'
    modelPath=os.path.sep.join([basePath,outputBase,'locator.h5'])
    plotPath=os.path.sep.join([basePath,outputBase,'plot.png'])
    testNames=os.path.sep.join([basePath,outputBase,'testImages.txt'])
    
    
    # Training Params cont
    # # earlyCallback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=delta,patience=tolerance,restore_best_weights=True,mode='auto')
    # # earlyCallback=tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=delta, patience=tolerance,restore_best_weights=True,mode='auto')
   
    # # checkpoint=tf.keras.callbacks.ModelCheckpoint(os.path.sep.join([basePath,outputBase,'Checkpoint.h5']),save_best_only=True,period=5)
    # # fitArgs={'callbacks':[earlyCallback,checkpoint],'verbose':1,'batch_size':batchSize,'epochs':noEpochs}
    # # fitArgs={'callbacks':[earlyCallback],'verbose':1,'batch_size':batchSize,'epochs':noEpochs}

    fitArgs={'verbose':1,'epochs':noEpochs,'batch_size':batchSize}
    
    
    
    
 
    
    
    
    
    #open labeled data
    rows=open(annotPath).read().strip().split("\n")
    rows=rows[1:]
    
    #initialize variables
    data = []
    targets = []
    filenames = []
    
    #load labeled data for every image
    for row in rows:
        #separate labeled data into image filename and 4 coordinates, read image 
        (fileName, sX, sY, eX, eY) = row.split(",")
        indivImP=os.path.sep.join([imPath,fileName])
        im = cv2.imread(indivImP)
       
        
        #load image for network input, reformat, get shape, convert to np array
        im = tf.keras.preprocessing.image.load_img(indivImP,target_size=targetShape)
        im = tf.keras.preprocessing.image.img_to_array(im)
        im=np.expand_dims(im,-1)
        shp=im.shape
        im=np.array(im)
        
        #append appropriate information to input / output variables
        data.append(im)
        targets.append((sX,sY,eX,eY))
        #targets.append([[sX],[sY],[eX],[eY]])
        filenames.append(fileName)
        
   
    #normalize input, convert to approrpriate datatype 
    data=np.array(data,dtype='float32')/255.00
    targets=np.array(targets,dtype='float32')
    
    #Model building: ResNet50V2 as base with dense layers before output
    resNet= tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    flat=tf.keras.layers.Flatten()(resNet.output)
    output1=tf.keras.layers.Dense(128,activation=acFunc,use_bias=True)(flat)
    output2=tf.keras.layers.Dense(64,activation=acFunc,use_bias=True)(output1)
    output3=tf.keras.layers.Dense(4,activation='linear',use_bias=True)(output2)
    cnn=tf.keras.models.Model(inputs=resNet.input,outputs=output3)
 
    #compile network and display summary
    cnn.compile(optimizer=optimize,loss=lossfxn,metrics=['accuracy'])  
    cnn.summary()

    #separate labeled data into test and training inputs/outputs
    x_train, x_test, y_train, y_test=train_test_split(data,targets,test_size=0.20,random_state=1738)

    
    #Train network
    hist=cnn.fit(x=x_train,y=y_train,**fitArgs) 
    
    #display loss and accuracy plots
    plt.plot(hist.history['loss'])
    plt.ylim([0,10])
    plt.show()
    plt.plot(hist.history['accuracy'])
    plt.show()

    
    

    #Display testing labels, predictions, and differences
    print('\nTesting Data\n')
    pred=cnn.predict(x_test)
    print(y_test)
    print(pred)
    print(y_test-pred)
    cnn.evaluate(x_test,y_test)
    
    #Display training labels, predictions, and differences
    print('\nTraining Data\n')
    pred=cnn.predict(x_train)
    print(pred)
    print(y_train)
    print(y_train-pred)
    

    
   #iteratie through testing or training images and plot labels vs predictions
    for image, coords, targ in zip(x_test, pred, y_test):
    #for image, coords, targ in zip(x_train, pred, y_train):
        #print(coords)
        sX,sY,eX,eY=coords;
        sX2,sY2,eX2,eY2=targ;
        
        image=image[:,:,:,0]
        
        sX=sX*(225/5808)
        eX=eX*(225/5808)
        
        sY=sY*(225/4124)
        eY=eY*(225/4124)
        
        sX2=int(sX2*(225/5808))
        eX2=int(eX2*(225/5808))
        
        sY2=int(sY2*(225/4124))
        eY2=int(eY2*(225/4124))
        
        sX,sY,eX,eY=int(sX),int(sY),int(eX),int(eY)    
    
    
        cv2.rectangle(image,(sX,sY),(eX,eY),(0,0,255),2)
        cv2.rectangle(image,(sX2,sY2),(eX2,eY2),(255,0,0),2)
        plt.figure()
        plt.imshow(image)
    
   