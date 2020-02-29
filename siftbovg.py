import os
import time
import cv2
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_recall_fscore_support


sift = cv2.xfeatures2d.SIFT_create()

def getImageMetadata(folderPath):
    training_names = os.listdir(folderPath)
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(folderPath, training_name)
        class_path = [os.path.join(dir, f) for f in os.listdir(dir)]
        image_paths+=class_path
        image_classes+=[class_id]*len(class_path)
        class_id+=1
    return image_paths,image_classes


def preProcessImages(image_paths):
    descriptors= []
    for image_path in image_paths:
        im = cv2.imread(image_path)
        kp, des = sift.detectAndCompute(im,None)
        descriptors.append(des)
    return descriptors

def train(descriptors,image_classes,image_paths):  
    flann_params = dict(algorithm = 1, trees = 5)     
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    bow_extract  =cv2.BOWImgDescriptorExtractor(sift,matcher)
    bow_train = cv2.BOWKMeansTrainer(60)
    for des in descriptors:
        bow_train.add(des)
    voc = bow_train.cluster()
    bow_extract.setVocabulary( voc )
    traindata = []
    for imagepath in image_paths:
        featureset = getImagedata(sift,bow_extract,imagepath)
        traindata.extend(featureset)
    return traindata

def getImagedata(feature_det,bow_extract,path):
    im = cv2.imread(path)
    featureset = bow_extract.compute(im, feature_det.detect(im))
    return featureset   

if __name__ == '__main__':
    img_paths,img_classes= getImageMetadata("bltraindata")
    start_time = time.time()
    des=preProcessImages(img_paths)
    td=train(des,img_classes,img_paths)
    
    labels =  np.array(img_classes,np.float32).reshape(len(img_classes),1)
    sift_features = np.array(td,np.float32).reshape(len(td),-1)
    data_frame = np.hstack((sift_features,labels))
    np.random.seed(11)
    np.random.shuffle(data_frame)
    
    percentage = 80
    partition = int(len(sift_features)*percentage/100)
    x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
    y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()
#    clf = LinearSVC(C=100).fit(x_train, y_train) 
    clf = SVC(kernel = 'linear', C = 10000).fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    print("elapsed time: {}s".format(round(elapsed_time,2)))
    y_pred = clf.predict(x_test)
    
     #Applyin K-fold cross validation
    from sklearn.model_selection import cross_val_score
    accuracies= cross_val_score(estimator= clf, X= x_train, y= y_train, cv=10)
    accuracies.mean()
    accuracies.std()
    
    
    
    print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
    print('\n')
    print(precision_recall_fscore_support(y_test, y_pred, average='weighted'))

    sb = ['DCVoltageSource', 'ACVoltageSource ', 'CurrentSource', 'ContolledVoltageSource', 'ControlledCurrentSource', 'Lamp', 'Diode', 'Capacitor(EU)', 'Capacitor(US)', 
               'Resistor(EU)', 'Resistor(US)', 'Inductor', 'Transistor', 'Switch', 'Battery', 'EarthGround', 'ChassisGround']
    
    cm= confusion_matrix(y_test, y_pred) 
    print(cm)
    df_cm = pd.DataFrame(cm, index = [i for i in sb],
                  columns = [i for i in sb])
    sn.set(font_scale=0.8)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 14})# font size
    