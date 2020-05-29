# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:29:31 2020

@author: adnane
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

'''Example images used in the report'''

#filename = './Dataset/S07-33-MAX_Dm-Kr walking No 182 - NC14.lif - overview.tif'
filename = './Dataset/S04-04-MAX_Dm-Kr length No 193.lif - overview.tif'
#filename = './Dataset/S05-05-MAX_Dm-Kr mRNA B2 B3 seperate probes 3h Amp.lif - overview.tif'
#filename = './Dataset/S06-06-MAX_Dm-Kr mRNA B2 B3 seperate probes ON Amp.lif - overview.tif'
#filename = './Dataset/S01-34-MAX_Dm-Kr walking No 182.lif - overview embryo1.tif'


def load_image(filename):
    '''returns the image in the filename from the datatset'''
    img = cv2.imreadmulti(filename)
    img = img[1][0]
#    plt.imshow(img)
    return(img)

def plot(img, n, m, title = ''):
    plt.title(title)
    if n < img.shape[0] and m <img.shape[1]:
#        print('ok')
        plt.imshow(img[img.shape[0]//2-n: img.shape[0]//2+n, img.shape[1]//2-m: img.shape[1]//2+m])
        plt.show()
    else:
        plt.imshow(img)
        plt.show()

img = load_image(filename)


n = 200
m = 400

min_area = 50


plot(img,n,m, 'Original')

'''Initial global Otsu thresholding'''
th,G = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plot(G,n,m, 'Otsu')


'''Finding connected components to apply watershed'''
total_components, markers, stats, centroids = cv2.connectedComponentsWithStats(G, connectivity = 4 )
markers[G==0] = 0

plot(255-G,n,m, 'Inverted OTSU: the watershed landscape''')

from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from skimage.filters import threshold_multiotsu

'''Applying watershed'''
watershed_markers = watershed(255-G, markers=markers)

plot(mark_boundaries(img, watershed_markers), n , m , 'Watershed Boundaries')

'''Identify regions smaller than 0.1*min_area'''
relevant_components = set()
for label in range(1,stats.shape[0]):
    area = stats[label,4]
    if area>min_area*0.1:
        relevant_components.add(label)


'''Local thresholding on watershed regions'''
B = img.copy()
component_to_threshold_map = {}
for component in relevant_components:
    '''if necessary, a speed improvement is possible by limiting the search of 
    the mask to the boundary box defined by connected components stats'''
    mask= watershed_markers==component
    temp_img = B[mask]
    th,_ = cv2.threshold(temp_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    component_to_threshold_map[component] = th
    B[np.logical_and(mask, B<th)]=0

'''Remove regions smaller than 0.1*min_area'''
irelevant_components = set(range(1,stats.shape[0])) - relevant_components
irelevant_components = list(irelevant_components)

B[np.isin(watershed_markers, irelevant_components)] = 0


plot(B,n ,m , title = 'Local Thresholding')


'''We detect and fill smaller than average holes'''
B[B>0] = 255
hole_total_components, hole_markers, hole_stats, hole_centroids = cv2.connectedComponentsWithStats(255-B, connectivity = 4) 

avg_area = sum(hole_stats[:,4])/len(hole_stats[:,4])    

fill_components = []
for component in range(1,hole_stats.shape[0]):
    area = hole_stats[component,4]
    if area<avg_area:
        fill_components.append(component)

B[np.isin(hole_markers, fill_components)] = 255


plot(B,n,m,'Filling small holes''')

'''Getting the connected components of our current image B'''
B_total_components, B_markers, B_stats, B_centroids = cv2.connectedComponentsWithStats(B, connectivity = 4)

'''Croping the components bigger than the min_area'''


from collections import Counter

components_to_crop = set()
component_areas = Counter(B_markers.flatten())

for component in component_areas:
    if component == 0:
        continue
    if component_areas[component]>min_area:
        components_to_crop.add(component)

components_to_remove = set(component_areas.keys()) - components_to_crop
components_to_remove = list(components_to_remove)

B[np.isin(B_markers, components_to_remove)] = 0


plot(B,n,m,'Removing small components''')


example_plotted = False

box_to_image_map = {}

def get_solidity(img):
    '''Compute the solidity defined by the area divided by the convex hull'''
    B_copy = img.copy()
    contours, hierarchy = cv2.findContours(B_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = lambda x: len(x))
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area>0:
        solidity = float(area)/hull_area
    else:
        solidity = 0
    return(solidity)

'''Processing of the areas one by one to identify the cell areas and divide touching cells'''
for component in components_to_crop:
    '''defining the boundaries of the box'''
    box_features = tuple(B_stats[component, 0:4])
    y1,x1,y2,x2 = box_features

    B_c = B[x1:x1+x2,y1:y1+y2]
    I_c = img[x1:x1+x2,y1:y1+y2]
    
    '''Combining the normalized distance matrix and brightness matrix to form the S matrix'''
    C = 255-B_c
    
    D = cv2.distanceTransform(C,cv2.DIST_L2, 5 )
    D_norm = (D -np.min(D))/(np.max(D)-np.min(D))
    if np.isnan(D_norm[0,0]):
        continue
    I_norm = (I_c -np.min(I_c))/(np.max(I_c)-np.min(I_c))
    
    S = I_norm-D_norm
    
    
    '''this is what the paper uses for the number of thresholds but it gets too high in practice'''
    t = int(component_areas[component]/ min_area) 
    '''Therefore we keep the maximum number of thresholds of the multi-Otsu thresholding to 2'''
    t = min(t,2)
    
    '''Applying multi otsu thresholding for quantization of S'''
    thresholds = threshold_multiotsu(S, classes = t+1)
    Map = np.digitize(S, bins=thresholds) 
    
    '''Cleaning process iterating over the thresholds'''
    Map = t - Map +1
    L = np.zeros(Map.shape)
    L = L.astype(np.int8)
    for i in range (1,t+1):
        L [Map==i]=1
        p = i*min_area/(t+1)
        
        total_components_L, markers_L, stats_L, centroids_L = cv2.connectedComponentsWithStats(L, connectivity = 4)
        
        for component in range(1,stats_L.shape[0]):
            area = stats_L[component,4]
            if area<p:
                mask= markers_L==component
                Map[mask] = i+1
    
    
    '''Applying watershed on the local minima of quantized S'''
    W = watershed(Map, markers = None, watershed_line=True)
    
    '''If there is more than one cell, in other words more than one local minimum connected component,
    apply postprocessing fusion operation if it improves solidity'''
    if 0 in W:
        '''solidity of the full component without separation'''
        merged_solidity = get_solidity(B_c)
        solidity_list = []
        list_components = np.unique(W)
        max_comp = max(list_components)
        for component in range(1, max_comp + 1):
            if component ==0:
                continue
            B_temp = B_c.copy()
            B_temp[W != component] = 0
            
            solidity = get_solidity(B_temp)
            if solidity == 0:
                continue
            solidity_list.append(solidity)
        
        '''Average solidity of the separated  components '''
        average_solidity = sum(solidity_list)/len(solidity_list)
        min_solidity = min(solidity_list)
        total_area = np.count_nonzero(B_c)
        
        if merged_solidity<min_solidity or total_area > 10 * min_area:
            B_c[W == 0] = 0
        
        
    B_c[W == 0] = 0
    
    box_to_image_map[box_features] = B_c
    
    if not example_plotted:    
        plot(S,n,m,'Example of normalized S matrix for one local region')
        example_plotted = True

'''Put the image back together using the local regions in the result'''

result = np.zeros(img.shape)

for box in box_to_image_map:
    y1,x1,y2,x2 = box
    result[x1:x1+x2,y1:y1+y2] = np.maximum(box_to_image_map[box], result[x1:x1+x2,y1:y1+y2])
    

plt.imshow(result)


result = result.astype(np.uint8)
B = result.copy()


'''Second part of the post processing, trying to find a more convex cell shape using the local area's brightness
it is set to False as it is slow and doesn't bring significant improvement'''



if False:
    '''Redefine local regions with the result matrix '''
    total_components, markers, stats, centroids = cv2.connectedComponentsWithStats(result, connectivity = 4 )
    watershed_markers = watershed(255-result, markers=markers)
    for component in range(1,stats.shape[0]):
        '''if necessary, a speed improvement is possible by limiting the search of 
        the mask to the boundary box defined by connected components stats'''
        mask= watershed_markers==component
        temp_img = result.copy()
        temp_img[markers!=component] = 0
        th,temp_img2 = cv2.threshold(img[mask],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        temp_img2 = np.zeros(result.shape, np.uint8)
        temp_img2[np.logical_and(mask, img>th)] =255
        
        area_1 = stats[component, 4]
        area_2 = np.count_nonzero(temp_img2)
        if area_1<min_area and area_2<min_area:
            B[mask]=0
            continue
        if area_1<min_area and area_2>min_area:
            B[np.logical_and(mask, B<th)]=0
            B[np.logical_and(mask, B>=th)]=255
            continue
        if area_1>min_area and area_2<min_area:
            continue
        
    
    
        solidity_1 = get_solidity(temp_img)
        solidity_2 = get_solidity(temp_img2)
        if solidity_2>solidity_1:
            B[np.logical_and(mask, B<th)]=0
            B[np.logical_and(mask, B>=th)]=255        
    

img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
plt.imshow(B)
plot(B,n = 50, m=50, title = "Final segmentation result for Abdolhosseini et al's method 100*100")
plot(img,n = 50, m=50, title ='Comparison with the original image 100*100')

plot(B,n = 200, m=400, title = "Final segmentation result for Abdolhosseini et al's method 400*800")
plot(img,n = 200, m=400, title ='Comparison with the original image 400*800')
