
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import uuid

import aws_thread as aws
import detection.frecon as frecon

import socket
import pickle
import struct
import numpy as np

import imagezmq.imagezmq as imagezmq

# 1. Arguments Parser  ###################################################################################################

ap = argparse.ArgumentParser()


ap.add_argument("-s", "--server-ip", required=True,
                help="ip address of the server to which the client will connect")

#Optional Args
ap.add_argument("-r", "--record", default=0, help="record and upload the video to aws s3")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")

ap.add_argument("-v", "--video", help="path to the video file")

args = vars(ap.parse_args())

####################################################################################################
warnings.filterwarnings("ignore")

####################################################################################################

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
p_prototxt = 'detection/objects_detection/MobileNetSSD_deploy.prototxt'
p_model = 'detection/objects_detection/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(p_prototxt, p_model)

# initialize the consider set (class labels we care about and want
# to count), the object count dictionary, and the frame  dictionary
# CONSIDER = set(["dog", "person", "car"])
CONSIDER = set(["person"])
objCount = {obj: 0 for obj in CONSIDER}
frameDict = {}

def Detect(frame):
	(h, w) = frame.shape[:2]
    
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# reset the object count for each object in the CONSIDER set
	objCount = {obj: 0 for obj in CONSIDER}

	response = 0
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.2:
			# extract the index of the class label from the
			# detections
			idx = int(detections[0, 0, i, 1])

			# check to see if the predicted class is in the set of
			# classes that need to be considered
			if CLASSES[idx] in CONSIDER:
				response = 1
				# increment the count of the particular object
				# detected in the frame
				objCount[CLASSES[idx]] += 1

				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the bounding box around the detected object on
				# the frame
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(255, 0, 0), 2)

	# draw the object count on the frame
	label = ", ".join("{}: {}".format(obj, count) for (obj, count) in objCount.items())
	
	cv2.putText(frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

	return response

# 2. List ini ###################################################################################################
firstFrame = None
frame_list = []
ped_res_list = []
face_res_list = []
time_list = []

index_start = 0
index_end = 0
index_current = 0

# 2. Simple Server Stream ###################################################################################################

sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(args["server_ip"]), REQ_REP = True)
# 2. Simple Server Stream ###################################################################################################

def CaptureVideo(frames, t):
    global cv2, vs

    print('[INFO] Saving video clip START of length: ', len(frames))

    if args.get("video", None) is None:
        frame_width = frame.shape[0]
        frame_height = frame.shape[1]
    else:
        frame_width = int(vs.get(3))
        frame_height = int(vs.get(4))

    rand=str(uuid.uuid4())
    
    path = '{}.mp4'.format(rand)

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    
    for f in frames:
        out.write(f)
    
    out.release()
    print('[INFO] Saving video clip DONE of length: ',len(frames))
    aws.upload(path)

# 3. VIDEO VS STREAM ###################################################################################################

if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(0.5)
else:
    vs = cv2.VideoCapture(args["video"])

# 4. Face Detector Ini ####################################################################################################

d = args.get("detector", None)
e = args.get("embedding", None)
r = args.get("recognizer", None)
l = args.get("le", None)

print(d,e,r,l)

frecon.init()

# 5. MAIN ####################################################################################################

# loop over the frames of the video
while True:
    
    frame = vs.read()
    
    if args.get("video", None) is None:
        frame = frame      
    else: 
        frame = frame[1]
        frame = cv2.flip(frame, 1)
        frame = cv2.flip(frame, 0)

    frame_list.append(frame)

    # Break if end video
    if frame is None:
        break

    # 5.1 Detection ####################################################################################################
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=400)
    
    green = (0,255,0) # Movement detection color
    blue = (255, 0, 0) # Pedestrian detection color
    
    ped_res = 0

    ped_res = Detect(frame)
    
    # frecon.FaceRecognition(frame)

    # Draw Results ####################################################################################################
    
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    time_list.append(ts)
    
    # cv2.putText(frame, "Movement: {}".format(mov_res), (10, 20),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)

    # cv2.putText(frame, "Pedestrian: {}".format(ped_res), (10, 40),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 2)
    
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
    0.35, (0, 0, 255), 1)
    
    # 5.2 Choose if saves the video ###################################################################################################
    
    ped_res_list.append(ped_res)
    
    tolerance = 7

    if index_current > 3 :
        s = sum(ped_res_list[index_current-1:index_current-tolerance:-1])
        s2 = sum(ped_res_list[index_current:index_current-tolerance:-1])
        last_frames = (s == 0) 

        if ped_res == 0 and (last_frames) and args.get('record', None):
            if index_end - index_start > 10 :
                CaptureVideo(
                    ped_res_list[index_start:index_end:1], 
                    time_list[index_start])
            index_end = index_current
            index_start = index_current
            pass
        elif ped_res == 0 and not (last_frames): 
            sender.send_image('Detector', frame)
            index_end += 1
        if ped_res == 1 and not (last_frames):
            sender.send_image('Detector', frame)
            index_end += 1
        elif ped_res == 1 and (last_frames):
            index_end = index_current
            index_start = index_current
        else:
            pass

    index_current += 1

    # 6. Show Image Feed ###################################################################################################
    
    cv2.imshow("Security Feed", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    ####################################################################################################

####################################################################################################
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
