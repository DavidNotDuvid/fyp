import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os 
import seaborn as sb
from skimage.measure import label, regionprops, regionprops_table
import time

hostname = "jupyter.mbi.nus.edu.sg"

def analysis(file_name):
    cwd = os.getcwd()
    print(file_name)
    print(type(file_name))
    image = cv2.imread(cwd + '/src/experiments/12/out/prob/'+file_name)
    image[:, :, 2] = 0
    mask = cv2.imread(cwd + '/src/mask/Mask2.png',0)
    res = cv2.bitwise_and(image,image, mask = mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    label_img = label(gray)
    regions = regionprops(label_img)
    fig, ax = plt.subplots()
    ax.imshow(label_img, cmap=plt.cm.gray)
    print("starting loop")
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    props = regionprops_table(label_img, properties=('centroid',
                                                 'orientation',
                                                 'major_axis_length',
                                                 'minor_axis_length'))
    print('end')
    return props
