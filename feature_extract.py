# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 19:43:18 2016

@author: sravanthi
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
from skimage.feature import hog
from skimage.transform import resize
from skimage import io, data, color, exposure
def feature_extract(path):
    image = io.imread(path)
    image = color.rgb2gray(image)
    image_resized = resize(image, (256, 256))
    return hog(image_resized, orientations=8,pixels_per_cell=(16, 16), cells_per_block=(1, 1))
def to_string(x):
    final_str = ""
    for i in range(len(x)):
        final_str = final_str + str(x[i])+","
    return final_str[:-1]
index = 0
row = 0
images = pd.DataFrame(index=range(1000),columns=["photo_id","feature_vec"])


path = "/home/ec2-user/yelp/train_photos/"

if (os.path.exists(os.path.join("/home/ec2-user/yelp/","output.csv")))==False :
    f = open("/home/ec2-user/yelp/output.csv",'a')
    file = csv.writer(f,lineterminator='\n')
    file.writerow(["photo_id","feature_vec"])
    f.flush()
    f.close()

for filename in os.listdir(path):
    if filename.startswith("."):
        continue
    photo_id = int(filename.split(".")[0])
    feature = feature_extract(os.path.join(path,filename))
    
    images.ix[row,"photo_id"] = photo_id
    images.ix[row,"feature_vec"] = feature
    index = index + 1
    row = row +1
    if index % 1000 == 0:
        print "Records processed:",index
        row = 0
        images["feature_vec"] = images["feature_vec"].apply(to_string)
        f = open("/home/ec2-user/yelp/output.csv",'a+')
        
        images.to_csv(f,header = False)
        f.flush()
        f.close()
        images = pd.DataFrame(index=range(1000),columns=["photo_id","feature_vec"])
        
if row != 0:
    print "Rows to be written:",row
    images["feature_vec"] = images["feature_vec"].apply(to_string)
    f = open("/home/ec2-user/yelp/output.csv",'a+')
    images.to_csv(f,header=False)
    f.flush()
    f.close()

    
    
    
     
    
     






