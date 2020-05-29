# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:10:22 2020

@author: Zhuofan Yu
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA,FastICA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

flag = np.array([0,0,0,0,0,
                 0,0,0,0,0,
                 0,0,0,0,0,
                 0,0,0,0,0,
                 0,0,0,0,0,
                 0,0,0,0,0,
                 0,0,0,0,0,
                 0,0,1,1,0,
                 0,0,0,0,1,
                 1,1,1,0,1,
                 1,1,1,1,0,
                 1,1,1,0,0,
                 1,1,1,0,1,
                 1,1,1,1,1
                 ])

def getConfusion(x,y,num = 1):
    for i in range(num):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)
        svclassifier = SVC(C=5,kernel='linear')
        svclassifier.fit(X_train, y_train)
        #svclassifier.fit(x, y)
        y_pred = svclassifier.predict(X_test)
        #y_pred = svclassifier.predict(x)
        if i == 0:
            M = confusion_matrix(y_test,y_pred)
            #M = confusion_matrix(y,y_pred)
        else:
            M += confusion_matrix(y_test,y_pred) 
    #print(classification_report(y_test,y_pred))
    #print(y_pred)
    return M
    
def showIMG(id,matrix):
    if matrix == 'S':
        img = S[id].reshape(1024,2048)
    elif matrix == 'X':
        img = X[id].reshape(1024,2048)
    elif matrix == 'D':
        img = D.T[id].reshape(1024,2048)
    plt.imshow(img)   
    plt.show()      
    
def loadD(savepath):
    return np.load(savepath+"D.npy")

def loadX(savepath):
    return np.load(savepath+"X.npy")
    
def loadS(savepath):
    return np.load(savepath+"S.npy")

def load_filenames(path):
    return os.listdir(path)

def removeCentral(M):
    col = 2048
    threshold = 550
    for i in range(1024):
        M[:,i*col + threshold:(i+1)*2048] = 0
    return M

def normalization():
    for num in range(40):
        showIMG(num,'S')
        record = []
        for i in range(70):
            record.append(np.dot(S[num],D.T[i]))
        record = (record-min(record)) / (max(record)-min(record))
        plt.vlines(1, min(record), max(record), colors = "r", linestyles = "dashed")
        plt.vlines(8, min(record), max(record), colors = "r", linestyles = "dashed")
        plt.vlines(20, min(record), max(record), colors = "r", linestyles = "dashed")
        plt.vlines(36, min(record), max(record), colors = "r", linestyles = "dashed")
        plt.vlines(46, min(record), max(record), colors = "r", linestyles = "dashed")
        plt.vlines(66, min(record), max(record), colors = "r", linestyles = "dashed")
        plt.plot(record)
        plt.show()
        
def unifyMatrix(M):
    for i,row in enumerate(M):
        norm = 0.0
        for dim in row:
            norm += dim ** 2
        M[i] = M[i] / np.sqrt(norm)
    return M

def getPrecentage(M):
    for i in range(len(M)):
        num = sum(M[i])
        M[i] = M[i] * 100 / num
    return M

filepath = "./GeneExpression/"
savepath = "./Results/"
filenames = load_filenames(filepath)
D = loadD(savepath)
X = loadX(savepath)
S = loadS(savepath) 
#S = removeCentral(S)
I_lge = np.dot(S,D)
I_lge = unifyMatrix(I_lge)
I_lge = I_lge.T    # each row is feature vector of an image 
#normalization()
    
M = getConfusion(I_lge,flag,num=5000)

A = getPrecentage(M)
print(A)