
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

import detection.pedestrian as ped
import detection.basic as md
import aws_thread as aws
import detection.frecon as frecon

import socket
import pickle
import struct

# 1. Arguments Parser  ###################################################################################################

ap = argparse.ArgumentParser()

# Required Args

ap.add_argument("-fd", "--detector", required=True, 	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding", required=True,     help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-z", "--recognizer", required=True, 	help="path to model trained to recognize faces")

#Optional Args
ap.add_argument("-r", "--record", default=0, help="record and upload the video to aws s3")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-l", "--le", required=True, help="path to label encoder")
ap.add_argument("-v", "--video", help="path to the video file")

args = vars(ap.parse_args())

####################################################################################################
warnings.filterwarnings("ignore")

# 2. List ini ###################################################################################################
firstFrame = None
frame_list = []
ped_res_list = []
face_res_list = []
time_list = []

index_start = 0
index_end = 0
index_current = 0

# 2. List ini ###################################################################################################

clientsocket = socket.socket( socket.AF_INET, socket.SOCK_STREAM )

clientsocket.connect(('localhost', 8485))

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

def encodeFrame(frame):
  result, frame = cv2.imencode('.jpg', frame, encode_param)
  data = pickle.dumps(frame, 0)
  size = len(data)

#   print("{}: {}".format(img_counter, size))

  s_data = struct.pack(">L", size) + data

  return s_data
 
def StreamVideo(frame):
    print('stream video')
    s_data = encodeFrame(frame)

    clientsocket.sendall(s_data)
# Function responsable of writing the video and interacting with the AWS Service
def CaptureVideo(frames, t):
    global cv2, vs

    # print(len(l))
    
    # frame_width = int(vs.get(3))
    # frame_height = int(vs.get(4))

    # rand=str(uuid.uuid4())
    
    # path = '{}.mp4'.format(rand)

    # out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    
    # for f in frames:
        # out.write(f)
    
    # out.release()
    # aws.upload(path)

# 3. VIDEO VS STREAM ###################################################################################################

# Stream read
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(0.5)

# Video read
else:
    vs = cv2.VideoCapture(args["video"])

# 4. Face Detector Ini ####################################################################################################

d = args.get("detector", None)
e = args.get("embedding", None)
r = args.get("recognizer", None)
l = args.get("le", None)

print(d,e,r,l)

# frecon.init(d,e,r,l)

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
    frame = imutils.resize(frame, width=500)
    
    green = (0,255,0) # Movement detection color
    blue = (255, 0, 0) # Pedestrian detection color
    
    # Movement detection
    mov_res = md.MovementDetection(frame, color=green )

    ped_res = 0
    
    # if movement detected, detect pedestrian
    if mov_res:
        ped_res = ped.PedestrianDetectionVideo(frame, color=blue)
    
    # frecon.FaceRecognition(frame)

    # Draw Results ####################################################################################################
    
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    time_list.append(ts)
    
    cv2.putText(frame, "Movement: {}".format(mov_res), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
    cv2.putText(frame, "Pedestrian: {}".format(ped_res), (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
    0.35, (0, 0, 255), 1)
    
    # 5.2 Choose if saves the video ###################################################################################################
    
    ped_res_list.append(ped_res) # Cumulate responses from pedestrian detection
    
    tolerance = 7 # Tolerance to intermitent non pedestrian detections

    if index_current > 3:
        s = sum(ped_res_list[index_current-1:index_current-tolerance:-1])
        s2 = sum(ped_res_list[index_current:index_current-tolerance:-1])
        last_frames = (s == 0) 

        if ped_res == 0 and (last_frames):
            if index_end - index_start > 20 and args.get('record', None):
                CaptureVideo(ped_res_list[index_start:index_end:1], time_list[index_start])
                # CaptureVideo(signal_list[index_start:index_end:1])
            index_end = index_current
            index_start = index_current
            # print('{}: {} start({}) and end({}) sum = {} (0)'.format(index_current, ped_res, index_start, index_end, s))
            pass
        elif ped_res == 0 and not (last_frames): # el de ahora es 0 y los de antes almenos 1 era 1 OCCUPIED
            StreamVideo(frame)
            index_end += 1
            # print('{}: {} start({}) and end({}) sum = {} (1)'.format(index_current, ped_res, index_start, index_end, s))
        if ped_res == 1 and not (last_frames): # el de ahora es 1 y los de antes almenos 1 era 1 OCCUPIED
            StreamVideo(frame)
            index_end += 1
            # print('{}: {} start({}) and end({}) sum = {} (2)'.format(index_current, ped_res, index_start, index_end, s))
        elif ped_res == 1 and (last_frames):
            index_end = index_current
            index_start = index_current
            # print('{}: {} update start({}) and end({}) sum = {} (3)'.format(index_current, ped_res, index_start, index_end, s))
        else:
            # print('{}: {} start({}) and end({}) sum = {} (4)'.format(index_current, ped_res, index_start, index_end, s))
            pass

    index_current += 1

    # 6. Show Image Feed ###################################################################################################
    
    cv2.imshow("Security Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

    ####################################################################################################

####################################################################################################
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
