#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:28:33 2019

@author: zan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:01:44 2019

@author: zan
"""

import os
import cv2
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


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

    # Sift feature descriptor
    bow_train = cv2.BOWKMeansTrainer(20)
    sift = cv2.xfeatures2d.SIFT_create()
    flann_params = dict(algorithm = 1, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {}) 
    bow_extract = cv2.BOWImgDescriptorExtractor( sift , matcher )
    sift_descriptors = []
    descriptor=[]

    k=0
    for data in trainData:
        img = cv2.imread(data,0)
        resized_img = cv2.resize(img,(100,100),interpolation = cv2.INTER_CUBIC)
        gauss_img = cv2.GaussianBlur(resized_img,(9,9),0)
        th = cv2.adaptiveThreshold(gauss_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        		cv2.THRESH_BINARY_INV,11,2)
        cv2.imwrite('Manipulated_data/'+str(k)+'_0.jpg',th)
        kp, des = sift.detectAndCompute(th,None)
        sift_descriptors.append(des)
        rows,cols = resized_img.shape
        for i in [1,2,3]:
            M   = cv2.getRotationMatrix2D((cols/2,rows/2),i*90,1)
            dst = cv2.warpAffine(th,M,(cols,rows))
            kp, des = sift.detectAndCompute(dst,None)
            sift_descriptors.append(des)
            cv2.imwrite('Manipulated_data/'+str(k)+'_'+str(i)+'.jpg',dst)
            
        k = k+1
    
    
    manipulated_data=sorted(os.listdir("Manipulated_data"))
    for d in sift_descriptors:
        bow_train.add(d)
        voc = bow_train.cluster()
        bow_extract.setVocabulary(voc)
        for th in manipulated_data:
            im=cv2.imread(th,0)
            siftkp = sift.detect(im)
            bowsig = bow_extract.compute(im, siftkp)
            descriptor.append(bowsig)
        
        
    labels =  np.array(trainLabel,np.float32).reshape(len(trainLabel),1)
    sift_features = np.array(descriptor,np.float32).reshape(len(descriptor),-1)
    data_frame = np.hstack((sift_features,labels))
    np.random.shuffle(data_frame)
    
    percentage = 80
    partition = int(len(sift_features)*percentage/100)
    x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
    y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()
    clf = SVC(kernel = 'linear', C = 1).fit(x_train, y_train) 
    y_pred = clf.predict(x_test)
    
    print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
    print('\n')
    print(classification_report(y_test, y_pred))

    sb = ['DCVoltageSource', 'ACVoltageSource ', 'CurrentSource', 'ContolledVoltageSource', 'ControlledCurrentSource', 'Lamp', 'Diode', 'Capacitor(EU)', 'Capacitor(US)', 
               'Resistor(EU)', 'Resistor(US)', 'Inductor', 'Transistor', 'Switch', 'Battery', 'EarthGround', 'ChassisGround']
    
    cm= confusion_matrix(y_test, y_pred) 
    print(cm)
    df_cm = pd.DataFrame(cm, index = [i for i in sb],
                  columns = [i for i in sb])
    sn.set(font_scale=0.8)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 14})# font size
    