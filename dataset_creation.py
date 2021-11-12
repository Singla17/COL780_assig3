# -*- coding: utf-8 -*-
"""
Somanshu 2018EE10314
Lakshya  2018EE10222
"""
import cv2
import argparse
import os
import json
import numpy as np
from utils import IOU

def parse_args():
    parser = argparse.ArgumentParser(description='Getting the test data')
    parser.add_argument('-r', '--root', type=str, default='pedestrian_detection', required=True, help="Path to the dataset root directory")
    parser.add_argument('-t', '--train', type=str, default='test', required=True, help="Path to the test json")
    parser.add_argument('-op', '--out_pos', type=str, default='out', required=True, help="Path to the output json") 
    parser.add_argument('-on', '--out_neg', type=str, default='out', required=True, help="Path to the output json") 
    
    args = parser.parse_args()
    return args

def prepare_pos_data(root,train_json,out_path):
    f = open(train_json)
    test_files = json.load(f)
    images_names = test_files['images']
    annos = test_files['annotations']
    counter=0
    for name in images_names:
        img_id = name['id']
        boxes = list(filter(lambda img: img['image_id'] == img_id, annos))
        image = cv2.imread(os.path.join(root,name['file_name']))
        
        for box in boxes:
            bbox = box['bbox']
            (x,y,w,h) = bbox
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            roi=image[y:y+h,x:x+w]
            cv2.imwrite(out_path+"/"+str(counter)+".png",roi)
            counter+=1
            
def getThreshold(wind,boxes):
    t=0
    for box in boxes:
        bbox = box['bbox']
        (x,y,w,h) = bbox
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        bbox_gt=[x,y,x+w,y+h]
        bbox_pred=[wind[0],wind[1],wind[0]+ wind[2],wind[1]+wind[3]]
        t=max(t,IOU(bbox_gt,bbox_pred))
    return t
            
def getWindowSize(boxes):   
    orig_size=640*1280
    orig_w=640
    orig_h=1280
    for box in boxes:
        bbox = box['bbox']
        (x,y,w,h) = bbox
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        area=w*h
        
        min_2=min(area,orig_size)
        if(min_2==area):
            orig_size=area
            orig_w=w
            orig_h=h
    return (orig_w,orig_h)
            
def prepare_neg_data(root,train_json,out_path):
    f = open(train_json) 
    test_files = json.load(f)
    images_names = test_files['images']
    annos = test_files['annotations']
    counter=0
    for name in images_names:
        img_id = name['id']
        boxes = list(filter(lambda img: img['image_id'] == img_id, annos))
        image = cv2.imread(os.path.join(root,name['file_name']))
        stepSize=4
        windowSize=getWindowSize(boxes)
        img_count=0
        y_rng=list(range(0, image.shape[0]-windowSize[1], stepSize))
        x_rng= range(0, image.shape[1]-windowSize[0], stepSize)
        itr=0
        while(img_count<10 and itr<20):
            x=np.random.choice(x_rng)
            y=np.random.choice(y_rng)
            itr+=1
            roi=image[y:y + windowSize[1], x:x + windowSize[0]]
            wind=[x,y,windowSize[0],windowSize[1]]
            T=0.25
            if(getThreshold(wind,boxes)<T):
                cv2.imwrite(out_path+"/"+str(counter)+".png",roi)
                counter+=1
                img_count+=1
    return
                        
       
if __name__ == "__main__":
    args = parse_args()
    root = args.root
    test_json = args.train
    output_path_pos = args.out_pos
    output_path_neg=args.out_neg
    prepare_neg_data(root, test_json, output_path_neg)
    prepare_pos_data(root, test_json, output_path_pos)

"""
root="./pedestrian_detection"
out_path="./pedestrian_detection/PennFudanPed/Output_Custom_HOG/training_data" 
train_json="./pedestrian_detection/PennFudanPed_full.json"
"""

"""
--root="./pedestrian_detection" --out_pos="./pedestrian_detection/PennFudanPed/Output_Custom_HOG/training_data_pos" --out_neg="./pedestrian_detection/PennFudanPed/Output_Custom_HOG/training_data_neg" --train="./pedestrian_detection/PennFudanPed_train.json"
"""
