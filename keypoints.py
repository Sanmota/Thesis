#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:34:59 2019

@author: zan
"""

import matplotlib.pyplot as plt
import cv2
#from skimage import data, exposure


image = cv2.imread('Manipulated_data/388.jpg',-1)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(image, None)
img_kp = cv2.drawKeypoints(image, kp, None)

plt.figure(figsize=(15, 15))
plt.imshow(img_kp); plt.show()
plt.xticks([]),plt.yticks([])