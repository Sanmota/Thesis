#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:01:44 2019

@author: zan
"""

import os
import time
import cv2
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.svm import SVC
<<<<<<< HEAD:multiclass_svm.py
=======
from sklearn.svm import LinearSVC
>>>>>>> e97d393df230a6ab94ed9487eae06e7592b4e22d:hog_svm.py
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_recall_fscore_support

def get_hog() :
    winSize = (100,100)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog

def load_trainData(image_path):
    trainData = []
    trainLabel = []
    i = 0
    for path in sorted(os.listdir(image_path)):

        files = os.listdir(image_path +'/'+ path)
        for f in files:
             if (f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.jpg')):
                 trainData.append(image_path +'/'+ path +'/' + f)
                 trainLabel.extend([x for x in np.repeat(i,4)])

        print ("{} -> {} ".format(path,i))
        i = i+1

    return trainData,trainLabel


if __name__ == '__main__':

    image_path = "Datasets"
    trainData,trainLabel  = load_trainData(image_path)

    # HoG feature descriptor
    hog = get_hog();
    hog_descriptors = []

    start_time = time.time()
    k=0
    for data in trainData:
        img = cv2.imread(data,0)
        resized_img = cv2.resize(img,(100,100),interpolation = cv2.INTER_CUBIC)
        gauss_img = cv2.GaussianBlur(resized_img,(9,9),0)
        th = cv2.adaptiveThreshold(gauss_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        		cv2.THRESH_BINARY_INV,11,2)
        cv2.imwrite('Manipulated_data/'+str(k)+'.jpg',th)
        hog_descriptors.append(hog.compute(th))
        rows,cols = resized_img.shape
        for i in [1,2,3]:
            M   = cv2.getRotationMatrix2D((cols/2,rows/2),i*90,1)
            dst = cv2.warpAffine(th,M,(cols,rows))
            hog_descriptors.append(hog.compute(dst))
            cv2.imwrite('Manipulated_data/'+str(k)+'_'+str(i)+'.jpg',dst)

        k = k+1

    #Building data frame from resulted features
    labels =  np.array(trainLabel,np.float32).reshape(len(trainLabel),1)
    hog_features = np.array(hog_descriptors,np.float32).reshape(len(hog_descriptors),-1)
    data_frame = np.hstack((hog_features,labels))
#    np.random.seed(11)
    np.random.shuffle(data_frame)
    
<<<<<<< HEAD:multiclass_svm.py
    
=======
    #Classification   
>>>>>>> e97d393df230a6ab94ed9487eae06e7592b4e22d:hog_svm.py
    percentage = 80
    partition = int(len(hog_features)*percentage/100)
    x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
    y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()
<<<<<<< HEAD:multiclass_svm.py
    clf = SVC(kernel = 'linear', C = 7500).fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    print("elapsed time: {}s".format(round(elapsed_time,2)))
=======
    clf = SVC(kernel = 'linear', C = 100).fit(x_train, y_train)
    #clf = LinearSVC(C=100).fit(x_train, y_train) 
    
    
    elapsed_time = time.time() - start_time
    print("elapsed time: {}s".format(round(elapsed_time,2)))
    
>>>>>>> e97d393df230a6ab94ed9487eae06e7592b4e22d:hog_svm.py
    y_pred = clf.predict(x_test)
    
    #Applyin K-fold cross validation
    from sklearn.model_selection import cross_val_score
    accuracies= cross_val_score(estimator= clf, X= x_train, y= y_train, cv=10)
    accuracies.mean()
    accuracies.std()
    
<<<<<<< HEAD:multiclass_svm.py
    
        
#    # Applying Grid Search to find the best model and the best parameters
#    from sklearn.model_selection import GridSearchCV
#    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#    grid_search = GridSearchCV(estimator = clf,
#                               param_grid = parameters,
#                               scoring = 'accuracy',
#                               cv = 10,
#                               n_jobs = -1)
#    grid_search = grid_search.fit(x_train, y_train)
#    best_accuracy = grid_search.best_score_
#    best_parameters = grid_search.best_params_
=======
>>>>>>> e97d393df230a6ab94ed9487eae06e7592b4e22d:hog_svm.py


    print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
    print('\n')
    print(precision_recall_fscore_support(y_test, y_pred, average='micro'))
    print(precision_recall_fscore_support(y_test, y_pred, average='macro'))
    print(classification_report(y_test,y_pred))
    sb = ['DCVoltageSource', 'ACVoltageSource ', 'CurrentSource', 'ContolledVoltageSource', 'ControlledCurrentSource', 'Lamp', 'Diode', 'Capacitor(EU)', 'Capacitor(US)', 
               'Resistor(EU)', 'Resistor(US)', 'Inductor', 'Transistor', 'Switch', 'Battery', 'EarthGround', 'ChassisGround']
    
    cm= confusion_matrix(y_test, y_pred) 
    print(cm)
    df_cm = pd.DataFrame(cm, index = [i for i in sb],
                  columns = [i for i in sb])
    sn.set(font_scale=0.8)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 14})# font size
    