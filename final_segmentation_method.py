# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:29:31 2020

@author: adnane
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

'''Example images used in the report'''


def calculate_average_size(filename, plot=False,n_image = 200, m_image =400, n_counting =100, m_counting = 100 ):
    '''Segmentation et computation of average size of a cell using the segmentation and the cell counting
    n_image and m_image are the dimensons of the processed window for segmentation
    n_counting and m_counting are the dimensions of the window for computing the average size'''
    
    
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m+h
    
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
            
    
    # =============================================================================
    # Step 1: Segementation
    # =============================================================================
    
    img = load_image(filename)
    original = img.copy()
    
    
    min_area = 50
    
    '''Initial median Blur to remove noise'''
    img = cv2.medianBlur(img,7)
    
    
    n = n_image
    m = m_image
    
    th = 20
    
    original = original[img.shape[0]//2-n: img.shape[0]//2+n, img.shape[1]//2-m: img.shape[1]//2+m]
    img = img[img.shape[0]//2-n: img.shape[0]//2+n, img.shape[1]//2-m: img.shape[1]//2+m]
    mask = np.where(img<th)
    
    
    if plot:
        plot(img,n,m, 'Blurred Original')
    
    '''Mean Adaptive Thresholding before watershed'''
    G = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,0)
    #th,G = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    if plot:
        plot(G,n,m, 'Adaptive mean thresholding')
    G[mask] = 0
    
    
    
    '''Watershed on connected components'''
    total_components, markers, stats, centroids = cv2.connectedComponentsWithStats(G, connectivity =8 )
    markers[G==0] = 0
    
    from skimage.segmentation import watershed
    from skimage.segmentation import mark_boundaries
    
    watershed_markers = watershed(255-G, markers=markers)
    
    if plot:
        plot(mark_boundaries(original, watershed_markers), n , m , 'Watershed Boundaries 500*800')
        plot(mark_boundaries(original, watershed_markers),100 , 100 , 'Watershed Boundaries 100*100')
    
    r = mark_boundaries(img, watershed_markers, color = -1)
    r = r[:,:,0]
    
    
    
    res = np.ones(original.shape)
    
    '''Computing the local thresholds using the watershed boundaries'''
    background_samples_list = []
    for i in range(1, total_components):
        '''Extracting pixels on the watershed lines as background samples'''
        background_samples = original[np.logical_and(r == -1, watershed_markers == i)]
        background_samples_list.append(background_samples)
        
        '''Calculate the 95% percentile of this sample and use it as a threshold'''
        max_percentile = np.percentile(background_samples, [95])[0]
        res [np.logical_and(original <= max_percentile, watershed_markers == i)] = 0
        
    original_median = original 
    
    original_median = cv2.cvtColor(original_median, cv2.COLOR_GRAY2RGB)
    
    res = res.astype(np.uint8)
    G_post = res.copy()
    G_post = G_post.astype(np.uint8)
    
    
    '''Postprocessing: Removing components smaller than the min area'''
    
    total_components, markers, stats, centroids = cv2.connectedComponentsWithStats(G_post, connectivity = 4)
    
    
    from collections import Counter
    
    components_to_crop = set()
    component_areas = Counter(markers.flatten())
    
    for component in component_areas:
        if component == 0:
            continue
        if component_areas[component]>min_area:
            components_to_crop.add(component)
    
    components_to_remove = set(component_areas.keys()) - components_to_crop
    components_to_remove = list(components_to_remove)
    
    G_post[np.isin(markers, components_to_remove)] = 0
    
    
    G_post[G_post>0] = 255
    
    '''Removing smaller than average black connected components'''
    hole_total_components, hole_markers, hole_stats, hole_centroids = cv2.connectedComponentsWithStats(255-G_post, connectivity = 4) 
    
    th_area = 10
    
    fill_components = []
    for component in range(1,hole_stats.shape[0]):
        area = hole_stats[component,4]
        if area<th_area:
            fill_components.append(component)
    
    G_post[np.isin(hole_markers, fill_components)] = 255
    
    if plot:
        plot(G_post,50,50, title = 'Final segmentation result for our method 100*100')
        plot(original_median , n =50, m= 50, title = 'Comparison with the original image 100*100')
        
        plot(G_post,200,400, title = 'Final segmentation result for our method 400*800')
        plot(original_median , n =200, m= 400, title = 'Comparison with the original image 400*800')
    
    
    # =============================================================================
    # Step 2 : Calculate the average size
    # =============================================================================
    
    '''Focus on an area in the center'''
    n= n_counting
    m = m_counting
    G_post = G_post[G_post.shape[0]//2-n: G_post.shape[0]//2+n, G_post.shape[1]//2-m: G_post.shape[1]//2+m]
    
    
    
    ''' removing components that touch the borders of the image'''
    total_components, markers, stats, centroids = cv2.connectedComponentsWithStats(G_post, connectivity = 4)
    
    borders = np.zeros(markers.shape)
    borders[0]=1
    borders[-1] = 1
    borders[: , 0] = 1
    borders[:, -1] = 1
    
    border_components = markers[np.logical_and(borders, markers)]
    border_components = np.unique(border_components)
    
    G_post[np.isin(markers,border_components)] = 0
    if plot:
        plot(G_post,100,100, title = 'Removing border components')
    
    total_cell_area = np.count_nonzero(G_post)
    print('total cell area', total_cell_area)
    
    '''counting cells '''
    sure_bg = G_post.copy()
    
    '''Distance transform'''
    dist_transform = cv2.distanceTransform(G_post,cv2.DIST_L2,5)
    
    if plot:
        plt.title('Distance transform')
        plt.imshow(dist_transform)
        plt.show()
    
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    '''Separate touching cells by watershed'''
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = markers.astype('int32')
    
    G_post = cv2.cvtColor(G_post, cv2.COLOR_GRAY2RGB)
    
    markers = cv2.watershed(G_post,markers)
    G_post[markers == -1] = [0,0,0]
    
    G_post = cv2.cvtColor(G_post, cv2.COLOR_BGR2GRAY)
    
    '''Median blur with bitwise and as post processing to remove tiny components created by watershed'''
    median = cv2.medianBlur(G_post,5)
    G_post = np.bitwise_and(median,G_post)
    
    if plot:
        plt.title('Final counted connected components')
        plt.imshow(G_post)
        plt.show()
    
    '''Final count of the connected components'''
    num_cells, markers, stats, centroids = cv2.connectedComponentsWithStats(G_post, connectivity =4 )
    
    min_area_count = 100
    
    '''Removing small components from the count'''
    for component in range(1,stats.shape[0]):
        area = stats[component,4]
        if area<min_area_count:
            num_cells-=1
    
    num_cells = num_cells - 1
    
    print('number of cells',num_cells )
    
    avg_size = total_cell_area / num_cells 
    print( 'average size',avg_size)
    
    return(avg_size)

if __name__ == "__main__":
    avg_size = calculate_average_size(filename)