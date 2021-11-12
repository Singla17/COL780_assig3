# Pedestrian Detection       
**Somanshu Singla(2018EE10314) and Lakshya Tangri(2018EE10222)**      

The Problem Statement can be found [here](./Problem_Statement.pdf) and the report can be found [here](./Report.pdf).

## File Structure 
1. HoG Based SVM 
   1. Pretrained Model from OpenCV: 
      1. eval_hog_pretrained.py:                                                                                        
         It contains all the code for running the pretrained HoG SVM Model, can be run as:         
          ```
         python eval_hog_pretrained.py --root <path to dataset root directory> -- test <path to test json> --out <path to output json>
        ```
   2. Custom HoG based SVM: 
      1.  dataset_creation.py: It is used to create the dataset for training the SVM 
      2.  SVM_train.py : based on the created dataset this script creates and saved the trained model
      3.  eval_hog_custom.py:         
   It contains the complete prediction pipline for the SVM model trained by us, can be run as:
        ```
         python eval_hog_custom.py --root <path to dataset root directory> -- test <path to test json> --out <path to output json> --model <path to trained SVM model>
        ```

2. Faster RCNN:
   1. Pretrained Model from Pytorch:
        1. dataset_class.py: This script implemenets the "Pedestrian_Dataset" class which helps to load the data to input to the model.
        2. eval_faster_rcnn.py: This script implements the complete inference pipeline, can be run as:
        ```
         python eval_faster_rcnn.py --root <path to dataset root directory> -- test <path to test json> --out <path to output json> --model <path to pretrained Faster RCNN weights file>
        ```
   
3. utils.py: Contains the utilities which have been used during the implementations.