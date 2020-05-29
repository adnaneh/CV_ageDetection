# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:31:33 2020

@author: Zhuofan Yu
"""

import cv2
import os
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt

def load_filenames(path):
    return os.listdir(path)



def getBorder(img):
    min_horizonal = 0
    max_horizonal = 0
    for i,line in enumerate(img):
        cnt = 0
        cnt += np.count_nonzero(line > 120)
        if cnt >= 50:
            if min_horizonal == 0:
                min_horizonal = i
            max_horizonal = i
    img_T = img.T
    min_v = 0
    max_v = 0
    for i,line in enumerate(img_T):
        cnt = 0
        cnt += np.count_nonzero(line > 50)
        if cnt >= 40:
            if min_v == 0:
                min_v = i
            max_v = i
    return [min_horizonal,max_horizonal,min_v,max_v]

def rescale_luminorsity(img_flatten):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = img_flatten
    alpha = 1.5
    beta = 15.0

    denoised = alpha * gray + beta
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    denoised = denoised * (denoised>80)

    #denoised = cv2.fastNlMeansDenoising(denoised, None, 31, 7, 21)
    return denoised

def get_shape(img):
#        filename = 'C:/Users/adnane/Desktop/Visord/Dataset/S01-34-MAX_Dm-Kr walking No 182.lif - overview embryo1.tif'
        th = 10
        mask = np.where(img<th)
        plt.imshow(img)
        img[mask]=0
        img = cv2.medianBlur(img,51)
#        plt.imshow(img)
        img[img!=0] = 255
        def undesired_objects (image):
            image = image.astype('uint8')
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
            sizes = stats[:, -1]
            max_label = 1
            max_size = sizes[1]
            for i in range(2, nb_components):
                if sizes[i] > max_size:
                    max_label = i
                    max_size = sizes[i]
            img2 = np.zeros(output.shape)
            img2[output == max_label] = 255
            return(img2)
        img = undesired_objects(img)
        return(img)
        
def reshape(img,img2,mask,standard_mask):
    #print(img.shape)
    #print(mask.shape)
    #print(standard_mask.shape)
    plt.imshow(mask)
    plt.show()
    for i,row in enumerate(standard_mask):
        standard_left,standard_right = getleftright(standard_mask[i])
        mask_left, mask_right = getleftright(mask[i])
        #print("standard:{} , mask:{}".format(standard_left,mask_left))
        if mask_right > mask_left and ((standard_right-standard_left)/(mask_right-mask_left)<2):

            row = img[i:i+15,mask_left:mask_right]
            row = cv2.resize(row, (int(standard_right-standard_left), 15), interpolation=cv2.INTER_AREA)
            img[i] = img[i] * 0
            img[i,standard_left:standard_right] = row[0]
            row = img2[i:i+15,mask_left:mask_right]
            row = cv2.resize(row, (int(standard_right-standard_left), 15), interpolation=cv2.INTER_AREA)
            img2[i] = img2[i] * 0
            img2[i,standard_left:standard_right] = row[0]
    return [img,img2]

def reshape2(img,img2,mask,standard_mask):
    #print(img.shape)
    #print(mask.shape)
    #print(standard_mask.shape)
    plt.imshow(mask)
    plt.show()
    for i in range((img.shape)[1]):
        standard_down,standard_up = getupdown(standard_mask[:,i])
        mask_down, mask_up = getupdown(mask[:,i])
        #print("standard:{} , mask:{}".format(standard_left,mask_left))
        if mask_up > mask_down and ((standard_up-standard_down)/(mask_up-mask_down)<2):

            col = img[mask_down:mask_up,i:i+2]
            col = cv2.resize(col, (2,int(standard_up-standard_down)), interpolation=cv2.INTER_AREA)
            img[:,i] = img[:,i] * 0
            img[standard_down:standard_up,i] = col[:,0]         
            col = img2[mask_down:mask_up,i:i+2]
            col = cv2.resize(col, (2,int(standard_up-standard_down)), interpolation=cv2.INTER_AREA)
            img2[:,i] = img2[:,i] * 0
            img2[standard_down:standard_up,i] = col[:,0]
    return [img,img2]
        
        
def getleftright(row):
    left = 5000
    right = -10
    #print(np.max(row))
    for j,pixel in enumerate(row):
        if pixel > 10:
            if j < left:
                left = j
            if j > right:
                right = j
    return [left,right]

def getupdown(col):
    down = 5000
    up = -10
    #print(np.max(row))
    for j,pixel in enumerate(col):
        if pixel > 10:
            if j < down:
                down = j
            if j > up:
                up = j
    return [down,up]
    
def preprocessImage(filename:str,standard_mask):
    imgs = cv2.imreadmulti(filename)
    mask = get_shape(imgs[1][0])
    mask_border = detect_border(mask)
    
    
    #plt.imshow(standard_mask)
    #plt.show()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs[1][0] = clahe.apply(imgs[1][0])
    #border = getBorder(imgs[1][0])
    img = imgs[1][1][mask_border[0]:mask_border[1],mask_border[2]:mask_border[3]]    # gene expression
    img3 = imgs[1][0][mask_border[0]:mask_border[1],mask_border[2]:mask_border[3]]   # microscropic images
    #mask = mask[mask_border[0]:mask_border[1],mask_border[2]:mask_border[3]] 
    #img2 = imgs[1][0]
 
    """
    #img = img * mask
    plt.imshow(img3)
    plt.show()
    
    bias = 40
    img = img[max(0,border[0]-bias):min(1023,border[1]+bias),max(0,border[2]-bias):min(2047,border[3]+bias)]
    img3 = img3[max(0,border[0]-bias):min(1023,border[1]+bias),max(0,border[2]-bias):min(2047,border[3]+bias)]
    img2[max(0,border[0]-5-bias):max(0,border[0]+5-bias),:] = 255
    img2[min(1023,border[1]+5+bias):min(1023,border[1]+5+bias),:] = 255
    img2[:,max(0,border[2]-5-bias):max(0,border[2]+5-bias)] = 255
    img2[:,min(2047,border[3]+5+bias):min(2047,border[3]+5+bias)] = 255
    """
    
    
    """img = rescale_luminorsity(img)"""
    
    img = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_CUBIC)
    img3 = cv2.resize(img3, (2048, 1024), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (2048, 1024), interpolation=cv2.INTER_CUBIC)
    img3,img = reshape2(img3,img,mask,standard_mask)
    #plt.imshow(img3)
    #plt.show()
    #plt.imshow(img)
    #plt.show()
    
 
    """
    a = 2
    img[:,0:400] = img[:,0:400] * float(a)
    img[img > 255] = 255
    img = np.round(img)
    img = img.astype(np.uint8)
    """
 

    return [img,img3]

 

def detect_border(img):
    left = 5000
    right = -5
    down = 5000
    up = -5
    for i,row in enumerate(img):
        for j,col in enumerate(img[i]):
            if img[i][j] != 0:
                if i <down :
                    down = i
                if i > up:
                    up = i
                if j < left:
                    left = j
                if j > right:
                    right = j
    return [down, up,left,right]

def preprocessImages():
    filepath = "./Dataset/"
    savepath3 = "./Embryos/"
    savepath = "./GeneExpression/"
    #savepath2 = "./Rectangles/"
    filenames = load_filenames(filepath)
    
    imgs = cv2.imreadmulti(filepath +filenames[0])
    mask = get_shape(imgs[1][0])
    mask_border = detect_border(mask)
    mask = mask[mask_border[0]:mask_border[1],mask_border[2]:mask_border[3]] 
    mask = cv2.resize(mask, (2048, 1024), interpolation=cv2.INTER_CUBIC)
    for filename in tqdm(filenames):
        img2save = preprocessImage(filepath + filename,mask)
        #cv2.imwrite(savepath+filename,img2save[0])
        #cv2.imwrite(savepath2+filename,img2save[1])
        #cv2.imwrite(savepath3+filename,img2save[1])
        
if __name__ == "__main__":
    preprocessImages()




