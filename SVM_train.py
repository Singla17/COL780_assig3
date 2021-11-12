# -*- coding: utf-8 -*-
"""
Somanshu 2018EE10314
Lakshya  2018EE10222
"""

import cv2
import argparse
import os
import numpy as np
import pickle
    
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def parse_args():
    parser = argparse.ArgumentParser(description='Training SVM')
    parser.add_argument('-o', '--out', type=str, default='test', required=True, help="Path to the test json")
    parser.add_argument('-i', '--inp_pos', type=str, default='out', required=True, help="Path to the output json") 
    parser.add_argument('-n', '--inp_neg', type=str, default='out', required=True, help="Path to the output json") 
    
    args = parser.parse_args()
    return args

def train(inp_pos,inp_neg,out_path):
    image_list = os.listdir(inp_pos)
    x=[]
    y=[]
    for img in image_list:
        frame=cv2.imread(os.path.join(inp_pos,img))
        frame=cv2.resize(frame,(240,120))
        fd= hog(frame, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), multichannel=True)
        x.append(fd)
        y.append(1)
    
    image_list = os.listdir(inp_neg)
   
    for img in image_list:
        frame=cv2.imread(os.path.join(inp_neg,img))
        frame=cv2.resize(frame,(240,120))
        fd = hog(frame, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), multichannel=True)
        x.append(fd)
        y.append(-1)
    x=np.array(x)
    y=np.array(y)
    print("data ready")
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto',verbose=True,probability=True))
    clf.fit(x, y)
    pkl_filename = out_path+"/customized_SVM.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
    print("model ready")

if __name__ == "__main__":
    args = parse_args()
    out_path = args.out
    inp_pos = args.inp_pos
    inp_neg=args.inp_neg
    train(inp_pos,inp_neg,out_path)
"""
root="./pedestrian_detection"
out_path="./pedestrian_detection/PennFudanPed/Output_Custom_HOG/training_data" 
train_json="./pedestrian_detection/PennFudanPed_full.json"
"""

"""
--inp_pos="./pedestrian_detection/PennFudanPed/Output_Custom_HOG/training_data_pos" --inp_neg="./pedestrian_detection/PennFudanPed/Output_Custom_HOG/training_data_neg" --out="./pedestrian_detection/PennFudanPed/Output_Custom_HOG"
"""

    