# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:17:27 2020

@author: adnane
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cv2 import adaptiveThreshold

def load_image(filename):
    '''returns the image in the filename from the datatset'''
    img = cv2.imreadmulti(filename)
    img = img[1][0]
    #plt.imshow(img)
    return(img)

def remove_zero_rows(img):
    inc = 0
    while np.all(img[inc]==0):
        inc+=1
    inc -=1

#filename = "C:/Users/adnane/Desktop/Dataset/S01-34-MAX_Dm-Kr walking No 182.lif - overview embryo1.tif"
#filename = 'C:/Users/adnane/Desktop/Dataset/S02-18-MAX_Dm-Kr walking No 179.lif - 19-12-06 - overview.tif'
#filename = './Dataset/S03-19-MAX_Dm-Kr walking No 179.lif - overview.tif'

def cell_counter(filename):
    img = load_image(filename)
    
    
    mask = np.where(img<50)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    #plt.imshow(img)
    
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,1)
    #plt.imshow(th2)
    
    th2[mask] = 0
    
    #plt.imshow(th2)
    
    median = cv2.medianBlur(th2,7)
    
    
    opening = median
    #blur = median
    #ret3,th4 = _, median
    #plt.imshow(opening)
    
    
    # sure background area
    sure_bg = median.copy()
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    #plt.imshow(dist_transform)
    
    ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
    #plt.imshow(sure_fg)
    #plt.imshow(sure_bg)
    
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = markers.astype('int32')
    
    #now load same image as color image
    img = cv2.cvtColor(median, cv2.COLOR_GRAY2RGB)
    
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [0,0,0]
    #plt.imshow(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, markers = cv2.connectedComponents(img)
    
    #print( str(ret) + ' cells')
    return ret
    
if __name__ == "__main__":
    filename = './Dataset/S03-19-MAX_Dm-Kr walking No 179.lif - overview.tif'
    calculateCells(filename)
"""
img2 = median [600:800, 600:800]
plt.imshow(img2)
"""

#import time
#ts = time.time()
#num = markers.max()
#N = 10
#
### If the count of pixels less than a threshold, then set pixels to `0`.
#for i in range(1, num+1):
#    pts =  np.where(markers == i)
#    
#    if len(pts[0]) < N:
#        print(len(pts[0]))
#        img[pts] = 0
#
#print("Time passed: {:.3f} ms".format(1000*(time.time()-ts)))



"""

img = img2
num_labels, labels_im = cv2.connectedComponents(img)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()
    return(labeled_img)

res = imshow_components(labels_im)
plt.imshow(res)

"""

"""
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np

n = 12
l = 256
np.random.seed(1)
im = np.zeros((l, l))
points = l * np.random.random((2, n ** 2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = img
blobs = im > 0.7 * im.mean()

all_labels = measure.label(blobs)
blobs_labels = measure.label(blobs, background=0)

plt.figure(figsize=(9, 3.5))
plt.subplot(131)
plt.imshow(blobs, cmap='gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(all_labels, cmap='nipy_spectral')
plt.axis('off')
plt.subplot(133)
plt.imshow(blobs_labels, cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()
"""