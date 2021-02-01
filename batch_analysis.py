import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import sys
from analysis import analysis


cwd = os.getcwd()
import_path = '/src/experiments/12/out/prob'
export_path = '/src/experiments/12/csv/'
def main():
    for i in range(300, 321):
        for j in range(20, 26):
            file_name = str('movie_2_nucleus_working_raw_t'+str(i)+"_z0" +str(j))
            print(file_name)
            temp = pd.DataFrame(analysis(file_name + '.png'))
            temp.to_csv(cwd + export_path + file_name +"_3", index = False, header=True)
    
    
if __name__ =="__main__":
    main() 