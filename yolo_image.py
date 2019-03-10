#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:22:51 2019

@author: alexdrake
"""

import numpy as np
import time
import cv2
import os

# default values
threshold = 0.3 # threshold when applying non-maxima suppression
min_confidence = 0.5 # minimum acceptable probability

# load the COCO class labels YOLO model was trained on
LABELS = open('yolo_cfg/coco_classes.txt').read().strip().split("\n")
 
# create a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# get the YOLO weights and model configuration
weightsPath = 'yolo_cfg/yolov3.weights'
configPath = 'yolo_cfg/yolov3.cfg'
 
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[Information] loading YOLO...")
yolo = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

included_extensions = ['jpg','jpeg', 'bmp', 'png'] # allowed extensions

# get list of files in the images directory
file_names = [fn for fn in os.listdir('input/')
              if any(fn.endswith(ext) for ext in included_extensions)]

print("[Information] Found {0} images".format(len(file_names)))

# apply YOLO detection to all image files
for file in file_names:
    print("[Information] Processing file {0} of {1}".format(file_names.index(file)+1,len(file_names)))
    image = cv2.imread('input/'+file)

    # load our input image and grab its spatial dimensions
    (H, W) = image.shape[:2]
     
    # determine only the output layer names that we need from YOLO
    op_layer_names = yolo.getLayerNames()
    op_layer_names = [op_layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]
     
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving bounding boxes and probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo.setInput(blob)
    start = time.time()
    layerOutputs = yolo.forward(op_layer_names)
    end = time.time()
     
    # show timing information on YOLO
    print("[Information] YOLO took {:.6f} seconds to complete".format(end - start))
    
    # initialize lists of detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
    	# loop over each of the detections
    	for detection in output:
    		# extract the class ID and confidence of the current object detection
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]
     
    		# filter out weak predictions where detected probability > minimum probability
    		if confidence > min_confidence:
    			# scale the bounding box coordinates back relative to the
    			# size of the image, keeping in mind that YOLO actually
    			# returns the center (x, y)-coordinates of the bounding
    			# box followed by the boxes' width and height
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")
     
    			# use the center (x, y)-coordinates to derive the top and
    			# and left corner of the bounding box
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))
     
    			# update our list of bounding box coordinates, confidences,
    			# and class IDs
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)
                
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, threshold)
    
    # make sure at least one detection exists
    if len(idxs) > 0:
    	# loop over the indexes we are keeping
    	for i in idxs.flatten():
    		# extract the bounding box coordinates
    		(x, y) = (boxes[i][0], boxes[i][1])
    		(w, h) = (boxes[i][2], boxes[i][3])
     
    		# draw a bounding box rectangle and label on the image
    		color = [int(c) for c in COLORS[classIDs[i]]]
    		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    			0.5, color, 2)
     
    # show the output image (for testing, disable for final)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    # save image in output folder
    cv2.imwrite('output/' + file.split('.')[0] + '.png', image) 