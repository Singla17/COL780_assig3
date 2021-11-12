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
import pickle
from skimage.feature import hog

def parse_args():
    parser = argparse.ArgumentParser(description='Getting the test data')
    parser.add_argument('-r', '--root', type=str, default='pedestrian_detection', required=True, help="Path to the dataset root directory")
    parser.add_argument('-t', '--test', type=str, default='test', required=True, help="Path to the test json")
    parser.add_argument('-o', '--out', type=str, default='out', required=True, help="Path to the output json")
    parser.add_argument('-m', '--model', type=str, default='out', required=True, help="Path to the model pkl")

    args = parser.parse_args()
    return args

def detectMultiScale(frame,pyr_len,winStride,windowSize,model):
    scaled_frames=[frame]
    boxes=[]
    scores=[]
    for pyr in range(pyr_len):
        frame=cv2.pyrDown(frame)
        scaled_frames.append(frame)
    
    for i in range(len( scaled_frames)):
        image=scaled_frames[i]
        scale=2**i
        
        
        for y in range(0, image.shape[0]-windowSize[1], winStride[0]):
            for x in range(0, image.shape[1]-windowSize[0], winStride[1]):
                roi=image[y:y + windowSize[1], x:x + windowSize[0]]
                fd = hog(roi, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), multichannel=True)
                pred_class=model.predict([fd])
                pred_prob=max(model.predict_proba([fd]))
                pred_prob=max(pred_prob)
                
                if(pred_class[0]==1):
                    boxes.append([x*scale,y*scale,windowSize[0]*scale,windowSize[1]*scale])
                    scores.append([pred_prob])
                
    boxes = np.array(boxes)
    scores = np.array(scores)           
    return (boxes,scores)  

def pedestrian_detection(root,test_json,output_path,show,show_path,model,verbose):
    with open(model, 'rb') as file:
        detector = pickle.load(file)
    
        f = open(test_json)
        test_files = json.load(f)
        images_names = test_files['images']
        results = []
    
    
        process_order = 0
        for name in images_names:
            image = cv2.imread(os.path.join(root,name['file_name']))
            image = cv2.bilateralFilter(image, 15, 75, 75)
            (boxes, scores) = detectMultiScale(image,0, (12, 24),(120,240),detector)
        
            for arr in boxes:
                arr[2]=arr[2]+arr[0]
                arr[3]=arr[3]+arr[1]
    
            #print("Before NMS "+ str(np.shape(boxes)))
            scores = np.array(scores)
            scores = scores.flatten()
            boxes_nms = non_max_suppression(boxes, probs=scores, overlapThresh=0.2)
            
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
            #print("After NMS "+str(np.shape(boxes)))
            
            #print(scores)
            
            result = {}
            result["boxes"]=boxes
            result["scores"]=scores
            result["image_id"]=name['id']
            process_order +=1 
            
            results.append(result)
            if verbose:
                print("Image " + str(process_order) +" is processed")
            
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
    model=args.model
    verbose = True
    show = True
    show_path = "./pedestrian_detection/PennFudanPed/Output_Custom_HOG/predicted_masks"
    pedestrian_detection(root, test_json, output_path, show, show_path,model,verbose)

"""
root = "./pedestrian_detection"
output_path = "./pedestrian_detection/PennFudanPed/Output_Custom_HOG/output.json"
test_json = "./pedestrian_detection/PennFudanPed_val.json"
show= True
show_path = "./pedestrian_detection/PennFudanPed/Output_Custom_HOG/predicted_masks"
model = "./pedestrian_detection/PennFudanPed/Output_Custom_HOG/customized_SVM.pkl"
"""

"""
--root="./pedestrian_detection"  --test="./pedestrian_detection/PennFudanPed_val.json" --out="./pedestrian_detection/PennFudanPed/Output_Custom_HOG/output.json" --model="./pedestrian_detection/PennFudanPed/Output_Custom_HOG/customized_SVM.pkl" 
"""