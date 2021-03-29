# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:35:42 2021

@author: giajordan
"""


import os
import tensorflow as tf
import matplotlib as plt
import cv2
#import sklearn
import numpy as np
from numba import cuda
import multiprocessing


physDevs=tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physDevs[0], enable=True)

#def train(cnn):
    
   



if __name__ == "__main__":
    noUnits=128
    noLayers=1
    acFunc='sigmoid'
    optimize='adam'
    lossfxn='mse'
    tolerance=6
    batchSize=4
    filtNo=3
    kSize=(70,70)
    noEpochs=10☺0
    unitsPerLayer=int(noUnits/noLayers)
    targetShape=(224,224)
    
    earlyCallback=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=tolerance,restore_best_weights=True,mode='min')
    fitArgs={'callbacks':earlyCallback,'verbose':1,'batch_size':batchSize,'epochs':noEpochs}
    
    
    
    
    basePath=r"C:\Users\giajordan\Documents\GitHub\LC-PreProcessing"
    imPath=os.path.sep.join([basePath,r'Images'])
    annotPath=os.path.sep.join([basePath,r'Data.csv'])
    
    
    outputBase=r'Output'
    modelPath=os.path.sep.join([basePath,outputBase,'locator.h5'])
    plotPath=os.path.sep.join([basePath,outputBase,'plot.png'])
    testNames=os.path.sep.join([basePath,outputBase,'testImages.txt'])
    
    
    
    
    
    
    rows=open(annotPath).read().strip().split("\n")
    rows=rows[1:]
    
    data = []
    targets = []
    filenames = []
    
    
    for row in rows:
        (fileName, sX, sY, eX, eY) = row.split(",")
        
        
        
        indivImP=os.path.sep.join([imPath,fileName])
        im = cv2.imread(indivImP)
       
        
        (h,w) = im.shape[:2]
        
        sX=float(sX)/w
        sY=float(sY)/h
        eX=float(eX)/w
        eY=float(eY)/h
        
            
        im = tf.keras.preprocessing.image.load_img(indivImP,target_size=targetShape)
        im = tf.keras.preprocessing.image.img_to_array(im)
        
        shp=im.shape
        
        im=np.array(im)
        
        data.append(im)
        targets.append((sX,sY,eX,eY))
        filenames.append(fileName)
        
    
    
    # cnn=tf.keras.Sequential()
    
    # for i in range(noLayers):
    #     #cnn.add(tf.keras.layers.Conv2D(
    #     #    (filtNo),kSize,input_shape=(4124,5808,3,),kernel_initializer='glorot_uniform',use_bias=True,padding='valid'
    #     #    ))
    #     cnn.add(tf.keras.layers.Conv2D(
    #         (filtNo),kSize,input_shape=(shp)))
    #     cnn.add(tf.keras.layers.MaxPool2D(pool_size=kSize,strides=(1,1),padding='valid'))
        
    # #cnn.summary()
    # cnn.add(tf.keras.layers.Dense(
    #      50,activation=acFunc,use_bias=False))   
    # #cnn.add(tf.keras.layers.Dense(
    # #     unitsPerLayer,activation=acFunc,use_bias=False))   
    # #cnn.add(tf.keras.layers.Dense(
    # #     unitsPerLayer/2,activation=acFunc,use_bias=False))   
    # #cnn.add(tf.keras.layers.Dense(
    # #     unitsPerLayer/4,activation=acFunc,use_bias=False))   
    # cnn.add(tf.keras.layers.Flatten())
    # cnn.add(tf.keras.layers.Dense(4,activation='sigmoid'))
    
      
        
    data=np.array(data,dtype='float32')/255.00
    targets=np.array(targets,dtype='float32')
    
    resNet= tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling=None)
    flat=tf.keras.layers.Flatten()(resNet.output)
    output1=tf.keras.layers.Dense(128,activation='relu')(flat)
    output2=tf.keras.layers.Dense(4,activation='linear')(output1)
    cnn=tf.keras.models.Model(inputs=resNet.input,outputs=output2)
 
        
    cnn.compile(optimizer=optimize,loss=lossfxn)  
    cnn.summary()
    
    #fitArgs.update(x=data,y=targets)
    #cnn.fit(x=data,y=targets,**fitArgs)
    
    
    cnn.fit(x=data,y=targets,**fitArgs) 
    
    #cnn.save(modelPath,save_format="h5")
    

    # p=multiprocessing.Process(target=train,args=(rtrnModel))
    # p.start()
    # p.join()
    
    
    
    targ=[sX,sY,eX,eY]
    indivImP=os.path.sep.join([imPath,fileName])
    im = cv2.imread(indivImP)
    im = tf.keras.preprocessing.image.load_img(indivImP,target_size=targetShape)
    im = tf.keras.preprocessing.image.img_to_array(im)
    im=np.array(im)
    im=np.expand_dims(im,axis=0)
    pred=cnn.predict(data)
    print(targets)
    print(pred)
    print(targets-pred)
    
    
    
