# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:35:42 2021

@author: giajordan
"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from math import floor, sqrt
from statistics import median, mean
import PIL as pil
from sklearn.model_selection import RepeatedKFold, cross_val_score, StratifiedKFold, train_test_split, StratifiedShuffleSplit
import keras_tuner as kt
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


physDevs=tf.config.list_physical_devices('GPU')


#build a model to search hyperparameter space
def buildHyperModel(hp):
    acFunc='sigmoid'
    lR=hp.Float('learning_rate',0.01,1.00,sampling='log')
    if hp.Choice('Optimizer',['Adam','Adagrad'],default='Adagrad')=='Adam':
        optimize=tf.keras.optimizers.Adam(learning_rate=lR)
    elif hp.Choice('Optimizer',['Adam','Adagrad'],default='Adagrad')=='Adagrad':
        optimize=tf.keras.optimizers.Adagrad(learning_rate=lR)
    #lossfxn='mse'
    lossfxn=hp.Choice('Loss',['mse','mean_absolute_percentage_error','mean_absolute_error'],default='mean_absolute_error')
    
    #if hp.Choice('Layers',[50, 101],default = 50) == 50:
    resNet= tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    #elif hp.Choice('Layers',[50, 101],default = 50) == 101:
        #resNet= tf.keras.applications.ResNet101V2(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    #elif hp.Choice('Layers',[50, 101, 152],default = 50) == 152:    
        #resNet= tf.keras.applications.ResNet152V2(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')   
    
    flat=tf.keras.layers.Flatten()(resNet.output)
    output1=tf.keras.layers.Dense(256,activation=acFunc,use_bias=True)(flat)
    output2=tf.keras.layers.Dense(128,activation=acFunc,use_bias=True)(output1)
    output3=tf.keras.layers.Dense(64,activation=acFunc,use_bias=True)(output2)
    output4=tf.keras.layers.Dense(32,activation=acFunc,use_bias=True)(output3)
    output5=tf.keras.layers.Dense(4,activation='sigmoid',use_bias=True)(output4)
    model=tf.keras.models.Model(inputs=resNet.input,outputs=output5)
     
    model.compile(optimizer=optimize,loss=lossfxn,metrics=['mean_absolute_error','mean_absolute_percentage_error'])
    
    return model

def buildModel():
    lR=0.00001
    optimize=tf.keras.optimizers.Adam(learning_rate=lR)
    lossfxn='mean_absolute_error'   
    acFunc='sigmoid'
    
    
    #Model building: ResNet50V2 as base with dense layers before output
    resNet= tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    flat=tf.keras.layers.Flatten()(resNet.output)
    output1=tf.keras.layers.Dense(256,activation=acFunc,use_bias=True)(flat)
    output2=tf.keras.layers.Dense(128,activation=acFunc,use_bias=True)(output1)
    output3=tf.keras.layers.Dense(64,activation=acFunc,use_bias=True)(output2)
    output4=tf.keras.layers.Dense(32,activation=acFunc,use_bias=True)(output3)
    output5=tf.keras.layers.Dense(4,activation='sigmoid',use_bias=True)(output4)
    model=tf.keras.models.Model(inputs=resNet.input,outputs=output5)
        
    #compile network and display summary
    model.compile(optimizer=optimize,loss=lossfxn,metrics=['mean_absolute_error','mean_absolute_percentage_error']) 
    model.summary() 

    return model



#bool, do hyperparameter search or no
hyperSearch=0

    
#Training Parameters
noEpochs=5000           #training epochs
targetShape=(224,224)   #Shape to load images as
tolerance=10            #early stopping epochs without improvement
delta=.0025             #minimum imrovement for early stopping 
testSize=0.20           #fraction of dataset to use for test split
batchSize=1             #batch size

#Location of repo
basePath=os.getcwd()
#pathing
imPath=os.path.sep.join([basePath,r'Images'])
annotPath=os.path.sep.join([basePath,r'Data.csv'])
outputBase=r'Output'
modelPath=os.path.sep.join([basePath,outputBase,'locatorOPT.h5'])
plotPath=os.path.sep.join([basePath,outputBase,'plot.png'])
testNames=os.path.sep.join([basePath,outputBase,'testImages.txt'])


# Training fit arguments
# # earlyCallback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=delta,patience=tolerance,restore_best_weights=True,mode='auto')
earlyCallback=tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=delta, patience=tolerance,restore_best_weights=True,mode='auto')
   
checkpoint=tf.keras.callbacks.ModelCheckpoint(os.path.sep.join([basePath,outputBase,'Checkpoint2.h5']),save_best_only=True,period=5)
# # fitArgs={'callbacks':[earlyCallback,checkpoint],'verbose':1,'batch_size':batchSize,'epochs':noEpochs}
#fitArgs={'callbacks':[earlyCallback],'verbose':1,'batch_size':batchSize,'epochs':noEpochs}

fitArgs={'verbose':1,'epochs':noEpochs,'batch_size':batchSize}

    

#open labeled data
rows=open(annotPath).read().strip().split("\n")
rows=rows[1:]

#initialize variables
data = []
targets = []
filenames = []
group=[]

#width and height of training images
w=5808
h=4124

#load labeled data for every image
for row in rows:
    #separate labeled data into image filename and 4 coordinates, read image 
    (fileName, sX, sY, eX, eY, g) = row.split(",")
    indivImP=os.path.sep.join([imPath,fileName])
    im = cv2.imread(indivImP)
   
    
    sX, sY, eX, eY,=float(sX),float(sY),float(eX),float(eY)
    
    #load image for network input, reformat, get shape, convert to np array
    im = tf.keras.preprocessing.image.load_img(indivImP,target_size=targetShape)
    im = tf.keras.preprocessing.image.img_to_array(im)
    im=np.expand_dims(im,-1)
    shp=im.shape
    im=np.array(im)
    
    #append appropriate information to input / output variables
    data.append(im)
    targets.append((sX/w,sY/h,eX/w,eY/h))
    group.append(g)
    filenames.append(fileName)
    
   
#normalize input, convert to approrpriate datatype 
data=np.array(data,dtype='float32')/255.00
targets=np.array(targets,dtype='float32')


if hyperSearch:
    tuner=kt.Hyperband(buildHyperModel, \
                          objective='mean_absolute_error', \
                          max_epochs=30, \
                          hyperband_iterations=1, \
                          directory="HP Search", \
                          seed=1738)

skf=StratifiedShuffleSplit(n_splits=2,test_size=testSize,random_state=17)
for train_index, val_index in skf.split(targets, group):
    print("TRAIN:", train_index, "TEST:", val_index)
    x_train=data[train_index,:,:,:,:]
    y_train=targets[train_index,:]
    
    x_val=data[val_index,:,:,:,:]
    y_val=targets[val_index,:]

if hyperSearch:
    tuner.search(x_train,y_train, \
            validation_data=[x_val,y_val], \
            epochs=30, \
            callbacks=[earlyCallback] \
                 )


    cnn=tuner.get_best_models()[0]
    goodParams=tuner.get_best_hyperparameters()[0]
        
    cnn.save(modelPath,save_format="h5")

#separate labeled data into test and training inputs/outputs
skf=StratifiedShuffleSplit(n_splits=4,test_size=testSize,random_state=1738)


hist=[]
pxError=np.empty([skf.get_n_splits(),1])

#train models on each fold and save
for i, [train_index, test_index] in enumerate(skf.split(targets, group)):
    print(i)
    
    #set model save path 
    modelPath=os.path.sep.join([basePath,outputBase,'locatorStrat'+str(i)+'.h5'])
    
    #split into this folds test and training splits
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train=data[train_index,:,:,:,:]
    y_train=targets[train_index,:]
    x_test=data[test_index,:,:,:,:]
    y_test=targets[test_index,:]
    
    ##For initial building
    # x_train=x_train[0:1,:,:,:,:]
    # y_train=y_train[0:1,:]
    
    # #Train network or Load
    # cnn=buildModel()
    # hist.append(cnn.fit(x=x_train,y=y_train,**fitArgs))
    # cnn.save(modelPath,save_format="h5")
    cnn=tf.keras.models.load_model(modelPath)
    
    
    
    # #display loss and accuracy plots
    # plt.plot(hist[i].history['loss'])
    # plt.ylim([0,10])
    # plt.show()
    # #plt.plot(hist.history['accuracy'])
    # plt.plot(hist[-1].history['mean_absolute_error'])
    # plt.show()
    # pxError[i]=hist[-1].history['mean_absolute_error'][-1]
    
    
    #Display training labels, predictions, and differences
    print('\nTraining Data\n')
    pred=cnn.predict(x_train)
    print(y_train)
    print(pred)
    print(y_train-pred)
    cnn.evaluate(x_train,y_train)
    
    for prediction,value in zip(pred,y_train):
        
        d1=sqrt(((prediction[0]*w-value[0]*w)**2)+((prediction[1]*h-value[1]*h)**2))
        d2=sqrt(((prediction[2]*w-value[2]*w)**2)+((prediction[3]*h-value[3]*h)**2))
        
        print(d1,d2)
        print(mean([d1,d2]))
    
    
    #Display testing labels, predictions, and differences
    print('\nTesting Data\n')
    pred=cnn.predict(x_test)
    print(y_test)
    print(pred)
    print(y_test-pred)
    cnn.evaluate(x_test,y_test)
    
    #calculate pixel error 
    for prediction,value in zip(pred,y_test):
        d1=sqrt(((prediction[0]*w-value[0]*w)**2)+((prediction[1]*h-value[1]*h)**2))
        d2=sqrt(((prediction[2]*w-value[2]*w)**2)+((prediction[3]*h-value[3]*h)**2))
        print(d1,d2)
        print(mean([d1,d2]))
    
    #write some metrics to a text file to ref later
    file=open("Performance.txt",'a')
    file.write(str(pred))
    file.write("\n")
    file.write(str(y_test-pred))
    file.write("\n\n")
    file.close()
    
    
     
    #iteratie through testing or training images and plot labels vs predictions
    for image, coords, targ in zip(x_test, pred, y_test):
    #for image, coords, targ in zip(x_train, pred, y_train):
          #print(coords)
          sX,sY,eX,eY=coords;
          sX2,sY2,eX2,eY2=targ;
         
          image=image[:,:,:,0]
         
          sX=sX*(225)
          eX=eX*(225)
         
          sY=sY*(225)
          eY=eY*(225)
         
          sX2=int(sX2*(225))
          eX2=int(eX2*(225))
         
          sY2=int(sY2*(225))
          eY2=int(eY2*(225))
         
          sX,sY,eX,eY=int(sX),int(sY),int(eX),int(eY)    
     
          plt.figure()
          cv2.rectangle(image,(sX,sY),(eX,eY),(0,0,255),2)
          cv2.rectangle(image,(sX2,sY2),(eX2,eY2),(255,0,0),1)
          plt.imshow(image)
     
            
    #print average error for different folds' models
    print("Average Error: " + str(pxError.mean()))
    file=open("Performance.txt",'a')
    file.write(str(pxError.mean()))
    file.close()
       