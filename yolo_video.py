#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:57:13 2019

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
op_layer_names = yolo.getLayerNames()
op_layer_names = [op_layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

included_extensions = ['mp4'] # allowed extensions

# get list of files in the images directory
file_names = [fn for fn in os.listdir('input/')
              if any(fn.endswith(ext) for ext in included_extensions)]

# we're going to take the first video file found otherwise it'll take forever!
file_names = file_names[0]

# initialize the video stream, pointer for output video file, and
# frame dimensions
vs = cv2.VideoCapture('input/'+file_names)
writer = None
(W, H) = (None, None)
 
# try to determine the total number of frames in the video file
try:
	prop = cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[Information] {} total frames in video".format(total))
 
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[Information] could not determine total # of frames")
	print("[Information] estimated completion time cannot be determined")
	total = -1
    
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
 
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
 
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
        
        	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	yolo.setInput(blob)
	start = time.time()
	layerOutputs = yolo.forward(op_layer_names)
	end = time.time()
 
	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
    
    	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
 
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > min_confidence:
				# scale the bounding box coordinates relative to the
    			# image size. Note YOLO returns the center (x, y)-coordinates of the bounding
    			# box followed by the width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
 
				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
 
				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, threshold)
 
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
 
			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		writer = cv2.VideoWriter('output/' + file_names.split('.')[0] + '.mp4', fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)
 
		# some information on processing single frame
		if total > 0:
			elapsed_time = (end - start)
			print("[Information] single frame took {:.4f} seconds".format(elapsed_time))
			print("[Information] estimated time to finish: {:.2f} seconds".format(elapsed_time * total))
 
	# write the output frame to disk (could also show here, but speedier to not do...)
	writer.write(frame)
 
# release the file pointers
print("[Information] releasing file pointers...")
writer.release()
vs.release()