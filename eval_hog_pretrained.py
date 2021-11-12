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
from imutils.object_detection import non_max_suppression
from utils import get_results_in_json
from scipy.special import softmax

def parse_args():
    parser = argparse.ArgumentParser(description='Getting the test data')
    parser.add_argument('-r', '--root', type=str, default='pedestrian_detection', required=True, help="Path to the dataset root directory")
    parser.add_argument('-t', '--test', type=str, default='test', required=True, help="Path to the test json")
    parser.add_argument('-o', '--out', type=str, default='out', required=True, help="Path to the output json")

    args = parser.parse_args()
    return args

def pedestrian_detection(root,test_json,output_path,show,show_path):
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    f = open(test_json)
    test_files = json.load(f)
    images_names = test_files['images']
    results = []
    
    
    process_order = 0
    for name in images_names:
        image = cv2.imread(os.path.join(root,name['file_name']))
        image = cv2.bilateralFilter(image, 15, 75, 75)
        (boxes, scores) = detector.detectMultiScale(image, winStride=(2, 2),padding=(12, 12), scale=1.05)
        
        for arr in boxes:
            arr[2]=arr[2]+arr[0]
            arr[3]=arr[3]+arr[1]

        #print(process_order)
        scores = np.array(scores)
        scores = scores.flatten()
        if np.shape(scores)[0]!=0:
            scores = softmax(scores)
        boxes_nms = non_max_suppression(boxes, probs=scores, overlapThresh=0.5)
        #print(scores)
        
        scores_nms = None
        First = True
        for box in boxes_nms:
            ind = np.where(np.all(boxes==box,axis=1))[0]
            if First:
                scores_nms=scores[ind]
                First = False
            else: 
                scores_nms = np.vstack((scores_nms,scores[ind]))
        
        boxes_nms = np.array(boxes_nms)
        if len(boxes_nms)!=0:
            boxes_nms[:,2] = boxes_nms[:,2]-boxes_nms[:,0]
            boxes_nms[:,3] = boxes_nms[:,3]-boxes_nms[:,1]
        boxes = boxes_nms
        scores = scores_nms
        boxes = boxes.astype(float)
        
        cond = np.where(scores<0.05)[0]
        #print(scores)
        scores = np.delete(scores,cond,axis=0)
        boxes = np.delete(boxes,cond,axis=0)
        #print(boxes)
        #print(boxes)
        #print(scores)
        
        
        result = {}
        result["boxes"]=boxes
        result["scores"]=scores
        result["image_id"]=name['id']
        process_order +=1 
        
        results.append(result)
        
        if show: 
            for (x, y, w, h) in boxes:
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            if not os.path.isdir(show_path):
                os.mkdir(show_path)
            
            cv2.imwrite(show_path+"/"+str(process_order)+".png",image)
      
    result_json = get_results_in_json(results)
    
    
    with open(output_path, 'w') as fout:
        json.dump(result_json, fout)
    
if __name__ == "__main__":
    
    args = parse_args()   
    root = args.root
    test_json = args.test
    output_path = args.out
    show = False
    show_path = "./pedestrian_detection/PennFudanPed/Output_Pretrained_HOG/predicted_masks"
    pedestrian_detection(root, test_json, output_path, show, show_path)
 
    
"""
root = "./pedestrian_detection"
output_path = "./pedestrian_detection/PennFudanPed/Output_Pretrained_HOG/output.json"
test_json = "./pedestrian_detection/PennFudanPed_val.json"
show= True
show_path = "./pedestrian_detection/PennFudanPed/Output_Pretrained_HOG/predicted_masks"
pedestrian_detection(root, test_json, output_path, show)
"""