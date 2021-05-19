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
#tf.config.experimental.set_memory_growth(physDevs[0], enable=True)

#def train(cnn):
    
   



if __name__ == "__main__":
    lR=1.0#0.005
    noUnits=128
    noLayers=1
    #acFunc='sigmoid'
    acFunc='sigmoid'
    #optimize=tf.keras.optimizers.Adam(learning_rate=lR)
    optimize=tf.keras.optimizers.Adagrad(learning_rate=lR)
    #lossfxn='mse'
    lossfxn='MeanAbsolutePercentageError'
    tolerance=15
    batchSize=25
    delta=.0005
    #filtNo=3
    #kSize=(35,35)
    
    #noEpochs=1000000
    noEpochs=7500
    unitsPerLayer=int(noUnits/noLayers)
    targetShape=(224,224)
    
    
    basePath=r"C:\Users\giajordan\Documents\GitHub\LC-PreProcessing"
    imPath=os.path.sep.join([basePath,r'Images'])
    annotPath=os.path.sep.join([basePath,r'Data.csv'])
    
    
    outputBase=r'Output'
    modelPath=os.path.sep.join([basePath,outputBase,'locator.h5'])
    plotPath=os.path.sep.join([basePath,outputBase,'plot.png'])
    testNames=os.path.sep.join([basePath,outputBase,'testImages.txt'])
    
    
    
    #earlyCallback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=delta,patience=tolerance,restore_best_weights=True,mode='auto')
    #earlyCallback=tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=delta, patience=tolerance,restore_best_weights=True,mode='auto')
   
    #checkpoint=tf.keras.callbacks.ModelCheckpoint(os.path.sep.join([basePath,outputBase,'Checkpoint.h5']),save_best_only=True,period=5)
    #fitArgs={'callbacks':[earlyCallback,checkpoint],'verbose':1,'batch_size':batchSize,'epochs':noEpochs}
    #fitArgs={'callbacks':[earlyCallback],'verbose':1,'batch_size':batchSize,'epochs':noEpochs}

    fitArgs={'verbose':1,'epochs':noEpochs,'batch_size':batchSize}
    
    
    
    
 
    
    
    
    
    
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
        
       
        
        
        # sX=(float(sX)/w)
        # sY=(float(sY)/h)
        # eX=(float(eX)/w)
        # eY=(float(eY)/h)
        
            
        #im = tf.keras.preprocessing.image.load_img(indivImP,target_size=targetShape,color_mode="grayscale")
        im = tf.keras.preprocessing.image.load_img(indivImP,target_size=targetShape)
        im = tf.keras.preprocessing.image.img_to_array(im)
        #im=im[:,:,0]
        im=np.expand_dims(im,-1)
        
        shp=im.shape
        
        im=np.array(im)
        
        
        data.append(im)
        targets.append((sX,sY,eX,eY))
        #targets.append([[sX],[sY],[eX],[eY]])
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
    
    
    resNet= tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    #resNet= tf.keras.applications.ResNet50V2(include_top=True,weights='imagenet',pooling='avg')
    flat=tf.keras.layers.Flatten()(resNet.output)
    output1=tf.keras.layers.Dense(128,activation=acFunc,use_bias=True)(flat)
    output2=tf.keras.layers.Dense(64,activation=acFunc,use_bias=True)(output1)
    output3=tf.keras.layers.Dense(4,activation='linear',use_bias=True)(output2)
    cnn=tf.keras.models.Model(inputs=resNet.input,outputs=output3)
 
        
    cnn.compile(optimizer=optimize,loss=lossfxn,metrics=['accuracy'])  
    cnn.summary()
    
    #fitArgs.update(x=data,y=targets)
    #cnn.fit(x=data,y=targets,**fitArgs)
    
    
    x_train, x_test, y_train, y_test=train_test_split(data,targets,test_size=0.20,random_state=1738)
    
    
    # x_train=data[0:2,:,:,:]
    # y_train=targets[0:2,:]
    
    #hist=cnn.fit(x=x_train,y=y_train,validation_split=0.30,**fitArgs) 
    hist=cnn.fit(x=x_train,y=y_train,**fitArgs) 
    
    plt.plot(hist.history['loss'])
    plt.ylim([0,10])
    plt.show()
    plt.plot(hist.history['accuracy'])
    plt.show()
    #cnn=tf.keras.models.load_model(modelPath)
    #cnn=tf.keras.models.load_model(os.path.sep.join([basePath,outputBase,r"Checkpoint - Copy.h5"]))
    #cnn.save(modelPath,save_format="h5")
    

    # p=multiprocessing.Process(target=train,args=(rtrnModel))
    # p.start()
    # p.join()
    
    
    
    # targ=[sX,sY,eX,eY]
    # indivImP=os.path.sep.join([imPath,fileName])
    # im = cv2.imread(indivImP)
    # im = tf.keras.preprocessing.image.load_img(indivImP,target_size=targetShape,color_mode='grayscale')
    # im = tf.keras.preprocessing.image.img_to_array(im)
    # im=np.array(im)
    # im=np.expand_dims(im,axis=0)
    
    print('\nTesting Data\n')
    pred=cnn.predict(x_test)
    print(y_test)
    print(pred)
    print(y_test-pred)
    cnn.evaluate(x_test,y_test)
    
    
    

    
   
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
    
    print('\nTraining Data\n')
    pred=cnn.predict(x_train)
    print(pred)
    print(y_train)
    print(y_train-pred)
    