# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:08:15 2020

@author: adnane
"""

'''Statistics on the images'''

from cell_counting import cell_counter
from final_segmentation_method import calculate_average_size
import os
import matplotlib.pyplot as plt

'''Make the list of files'''
folder = './Dataset'


def load_filenames(path):
    res = []
    for filename in os.listdir(path):
        res.append(path + '/' + filename)
    return res


filenames = load_filenames(folder)

'''Calculate the cell size and cell count for each image'''
stats = {}

columns = {'stage', 'cell_count', 'cell_size'}
for column in columns:
    stats[column] = []


for filename in filenames:
    print(filename)
    stage = filename.split('-')[0][-3:]
    stats['stage'].append(stage)
    cell_count = cell_counter(filename)
    stats['cell_count'].append(cell_count)

    cell_size = calculate_average_size(filename)
    stats['cell_size'].append(cell_size)

'''Calculate aggregates and plots'''


import pandas as pd
df = pd.DataFrame(stats)

stages_changes = []
stage_list = list(df.stage)
for i in range(1, len(stage_list)):
    if stage_list[i] != stage_list[i - 1]:
        stages_changes.append(i)

df.groupby('stage').cell_count.mean()
#
d = df.groupby('stage').cell_size.mean()

for change in stages_changes:
    plt.axvline(x=change, color='r', linestyle='--')
plt.title('Cell count distribution')
plt.plot(df.cell_count)
plt.show()

for change in stages_changes:
    plt.axvline(x=change, color='r', linestyle='--')
plt.title('Cell size distribution')
plt.plot(df.cell_size)
plt.show()

