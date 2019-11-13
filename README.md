# OpenCV / Pedestrian Detection / AWS S3 / SOSAFE Reports

This repo shows a proof of concept of survelliance script that uploads to AWS S3 and reports to SOSAFE Services.

The code is based on PyImageSearch blog

## Tech

* [Python 3.7.0] - Must be installed

## Setup
- Clone repo
- Create virtual environment
- Activate virtualenv
- Run: pip install -r requirements
- Create inside root folder a "config.json" with the following params: 

```json
{
	"ACCESS_KEY_ID" : "access_key_id",
	"ACCESS_SECRET_KEY" : "access_secret_key",
	"BUCKET_NAME" : "bucket_name",
	"SOSAFE_AUTH_TOKEN": "token"
}
```

## Considerations

## Run
Run the following code to open the camera:

```sh
python app.py --detector detection/face_detection_model --embedding detection/openface_nn4.small2.v1.t7 --recognizer detection/output/recognizer.pickle --le .\detection\output\le.pickle
```
Run the following code to open a video:

```sh
python app.py --detector detection/face_detection_model --embedding detection/openface_nn4.small2.v1.t7 --recognizer detection/output/recognizer.pickle --le .\detection\output\le.pickle -v video_path
```

To run and record to AWS S3 put the "-r 1" argument, as shown below:

```sh
python app.py --detector detection/face_detection_model --embedding detection/openface_nn4.small2.v1.t7 --recognizer detection/output/recognizer.pickle --le .\detection\output\le.pickle -v ..\petu_silvi.mp4 -r 1
```
## Face Detection
The face regocnition model was trainned to identify 3 labels:
- Joaquin: Single face photos of Joaquin
- Barbara: Single face photos of Barbara
- Unknow: Photos of different single face persons

#### Activate Face Detection
Currently the face detection and recognition is OFF

To activate it just uncomment the following code:

```py
frecon.init(d,e,r,l) # line 97
```

and

```py
frecon.FaceRecognition(frame) # line 135
```

## Code Sections: app.py

- 1 Argument parsing
- 2 Lists ini
- 3 Video vs Stream
- 4 Face detection ini
- 5 Main video capture
- 5.1 Detection
- 5.2 Video Recording

## Todos

 - Improve movement detection for camera use.
 - Improve face recognition models accuracy.
 - Try YOLO deeplearning framework for pedestrian recognition.
 - Separete logic into Client, Server and MicroServices.
 
----

License MIT

