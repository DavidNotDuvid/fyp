from __future__ import print_function
import cv2 
import argparse
from config import config_vars
import os
import utils.dirtools

cwd = os.getcwd()
path = cwd + "/src/norm_images/"
data_partitions = utils.dirtools.read_data_partitions(config_vars)
files = data_partitions["training"]
for file in files:
    file_path = path + file
    print(file_path)
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(img)
    print(dst.shape)
    cv2.imwrite(cwd +'/src/hist/'+ file,dst)