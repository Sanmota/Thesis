#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:14:49 2019

@author: zan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:57:30 2019

@author: zan
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Image is loaded with imread command 
img = cv2.imread('Circuits/93.jpg',0)

#Blurring to reduce some noise
im_b= cv2.GaussianBlur(img,(5,5),0)

#Applying thresholding 
th, im_th =cv2.threshold(im_b, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 
#Copy the thresholded image.
im_floodfill = im_th.copy()
 
#Mask used to flood filling.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
#Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
 
#Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

#Creating Kernel 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

#Closing for extraction closed symbols 
closing = cv2.morphologyEx(im_floodfill_inv, cv2.MORPH_CLOSE, kernel,iterations=3)

#Contouring for closed shape symbols
im_rec=img.copy()
contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
#Segmenting closed symbols    
i=0
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	if (w>50 and w<300) or (h>50 and h<300):
		cl_output= img[y-40:y+h+40,x-40:x+w+40]
		cv2.imwrite(os.path.join("Segmented_symbols/",str(i)+".jpg"), cl_output)
		i=i+1



#extracting remaining circuit
dilation = cv2.dilate(im_floodfill_inv, kernel, iterations=7)
im_thb= im_th.copy()
        
contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
sec=im_th-dilation
for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if (w>300 or h>300):
            sec= im_th
            
#blurring remaining circuit
im_b1= cv2.GaussianBlur(sec,(5,5),0)
#Applying thresholding 
th, im_th1 =cv2.threshold(im_b1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#Creating Kernel 
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

dilation1 = cv2.dilate(im_th1, kernel1, iterations=10)
erosion = cv2.erode(dilation1, kernel1, iterations=14)
result= cv2.dilate(erosion, kernel1, iterations=10)

#Contouring for closed symbols
contours, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#Segmenting closed symbols
i=0
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	if (w>50 and w<300) or (h>50 and h<300):
		output= img[y-40:y+h+40,x-40:x+w+40]
		cv2.imwrite(os.path.join("Segmented_symbols/",str(i+10)+".jpg"),output)
		i=i+1

titles = ['Original Image', "Thresholded Binary Image",
          'Inverted Floodfilled Image','Remained Circuit','Image after morphological operation']
images = [img, im_thb, im_floodfill_inv, sec, result]


for i in range(5):
    plt.subplot(3,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


