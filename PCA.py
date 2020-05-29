# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:20:19 2020

@author: Zhuofan Yu
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA,FastICA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from cell_counting import cell_counter
from imageAlignment import preprocessImages
from final_segmentation_method import calculate_average_size

def load_image(filename,layer=0):
    """returns the image in the filename from the dataset"""
    img = cv2.imreadmulti(filename)
    img = img[1][0]
    #plt.imshow(img)
    return img

def threshold_img(M):
    M = M * (M>0.005)
    return M
    
def load_image_processed(filename):
    """returns the image in the filename from the dataset"""
    img = cv2.imreadmulti(filename)
    img = img[1][0]
    #plt.imshow(img)
    return img

def img2dim1(img):
    return img.reshape(-1,1)

def load_filenames(path):
    return os.listdir(path)

def construct_matrix(filenames:list):
    filepath = "./Dataset/"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i,filename in enumerate(filenames):
        img = load_image(filepath+filename)
        
        img = clahe.apply(img)
        img = img2dim1(img)
        if i == 0:
            D = img
        else:
            D = np.concatenate((D,img),axis=1)
    return D

def construct_matrix_geneexpression(filenames:list):
    filepath = "./GeneExpression/"
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i,filename in enumerate(filenames):
        img = load_image_processed(filepath+filename)  
        #img = clahe.apply(img)
        img = img2dim1(img)
        if i == 0:
            D = img
        else:
            D = np.concatenate((D,img),axis=1)
    return D

def construct_matrix_embryos(filenames:list):
    filepath = "./Embryos/"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i,filename in enumerate(filenames):
        img = load_image_processed(filepath+filename)  
        img = clahe.apply(img)
        img = img2dim1(img)
        if i == 0:
            D = img
        else:
            D = np.concatenate((D,img),axis=1)
    return D

def getEigenEmbryos(D,threshold_ratio=1.0):
    pca = PCA()
    pca.fit_transform(D)
    new_ratios = pca.explained_variance_ratio_
    new_embryos = pca.components_
    embryos = []
    ratios = []
    for i,ratio in enumerate(new_ratios):
        ratios.append(ratio)
        embryos.append(new_embryos[i])
        if sum(ratios) >= threshold_ratio:
            break
    return [embryos,ratios]

"""def getLGEs(X):
    ica = FastICA()
    X_transpose = X.T
    u = ica.fit_transform(X_transpose)
    u = u.T
    return u"""
    
def getLGEs(X):
    ica = FastICA()
    X_transpose = X.T
    u = ica.fit_transform(X_transpose)
    u = u.T
    return u

def showIMG(id,matrix):
    if matrix == 'S':
        img = S[id].reshape(1024,2048)
    elif matrix == 'X':
        img = X[id].reshape(1024,2048)
    elif matrix == 'D':
        img = D.T[id].reshape(1024,2048)
    plt.imshow(img)   
    plt.show()      
    
def saveMatrix(savepath,D,X,S):
    np.save(savepath+"D.npy",D)
    np.save(savepath+"X.npy",X)
    np.save(savepath+"S.npy",S)

def loadD(savepath):
    return np.load(savepath+"D.npy")

def loadX(savepath):
    return np.load(savepath+"X.npy")
    
def loadS(savepath):
    return np.load(savepath+"S.npy")

def unifyVector(A):
    norm = 0.0
    for item in A:
        norm += item ** 2
    return A/np.sqrt(norm)

def unifyMatrix(M):
    for i,row in enumerate(M):
        norm = 0.0
        for dim in row:
            norm += dim ** 2
        M[i] = M[i] / np.sqrt(norm)
    return M

def loadLabels(filenames):
    labels = []
    for filename in filenames:
        labels.append(filename[2])
    labels = np.array(labels)
    return labels.reshape(-1,1)

def loadCellNumber(filenames):
    feature = []
    for filename in filenames:
        #feature.append(calculateCells("./Dataset/" + filename))
        feature.append(cell_counter("./Dataset/" + filename))                
    feature = pd.DataFrame(feature)
    feature = np.array(feature)
    return feature

def loadCellSize(filenames):
    feature = []
    for filename in filenames:
        feature.append(calculate_average_size("./Dataset/" + filename)) 
    feature = pd.DataFrame(feature)
    feature = np.array(feature)
    return feature

def loadFeatures():
    filepath = "./Dataset/"
    savepath = "./Results/"#load embryos
    filenames = load_filenames(filepath)
    labels = loadLabels(filenames)
    cellnum = loadCellNumber(filenames)
    #cellsize = loadCellSize(filenames)
    
    D = loadD(savepath)
    S = loadS(savepath) 
    I_lge = np.dot(S,D) # each column is feature vector of an image  
    I_lge = unifyMatrix(I_lge)
    I_lge = I_lge.T    # each row is feature vector of an image 
    
    savepath = "./ResultsEmbryos/"
    #D = loadD(savepath)
    #S = loadS(savepath)
    #X = loadX(savepath)
    #I = np.dot(S,D)
    #I = np.dot(X,D)
    #I = unifyMatrix(I)
    #I = I.T
    #print(I_lge.shape)
    #I_lge = np.concatenate((I_lge,I),axis=1)
    #print(I_lge.shape)
    #print(cellnum.shape)
    I_lge = np.concatenate((I_lge,cellnum),axis=1)
    #I_lge = np.concatenate((I_lge,cellsize),axis=1)
    I_lge = np.concatenate((I_lge,labels),axis=1)
    I_lge = pd.DataFrame(I_lge)
    last = (I_lge.shape)[1]-1
    x = I_lge.drop(last,axis=1)
    y = I_lge[last] 
    x = x.drop(0,axis=0)    #ignore stage = 1
    y= y.drop(0,axis=0)
    return x,y

def getConfusion(x,y,num = 1):
    for i in range(num):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)
        svclassifier = SVC(C=0.5,kernel='linear')
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        if i == 0:
            M = confusion_matrix(y_test,y_pred)
        else:
            M += confusion_matrix(y_test,y_pred) 
    #print(classification_report(y_test,y_pred))
    return M
def initialization():
    print("##############Start pre-process############")
    #preprocessImages()
    print("##############Finish pre-process############")
    #filepath = "./Dataset/"
    filepath = "./GeneExpression/"
    savepath = "./Results/"
    filenames = load_filenames(filepath)
    print("##############Start processing gene expression############")
    D = construct_matrix_geneexpression(filenames)
    components = getEigenEmbryos(D.T,threshold_ratio=0.9)
    X = components[0]       #M'-by-N matrix, collection of the M' Eigen-Embryos(each row is a Eigen-Embryo)
    X = np.array(X)
    S = getLGEs(X)
    #S = threshold_img(S)
    saveMatrix(savepath,D,X,S)
    print("##############Finish processing gene expression############")
    print("##############Start processing embryos############")
    filepath = "./Embryos/"
    savepath = "./ResultsEmbryos/"
    D = construct_matrix_embryos(filenames)
    components = getEigenEmbryos(D.T,threshold_ratio=0.9)
    X = components[0]       #M'-by-N matrix, collection of the M' Eigen-Embryos(each row is a Eigen-Embryo)
    X = np.array(X)
    S = getLGEs(X)
    #S = threshold_img(S)
    saveMatrix(savepath,D,X,S)
    print("##############Finish processing embryos############")
    
def getPrecentage(M):
    for i in range(len(M)):
        num = sum(M[i])
        M[i] = M[i] * 100 / num
    return M
        



if __name__ == "__main__":
    #initialization()
    x,y = loadFeatures()
    M = getConfusion(x,y,num=300)
    A = getPrecentage(M)
    print(A)

  