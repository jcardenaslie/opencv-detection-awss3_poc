# OpenCV / Pedestrian Detection / AWS S3 / SOSAFE Reports

This repo shows a proof of concept of survelliance script that uploads to AWS S3 and reports to SOSAFE Services.

The code is based on PyImageSearch blog

## Tech

* [Python 3.7.0] - Must be installed

## Setup
- Clone this repo
- Create virtual environment
- Activate virtualenv
- Run: pip install -r requirements
- Inside a different folder clone streaming receiver: https://github.com/jcardenaslie/zmq-receiver-streaming
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

## Run Server
- Create virtual environment
- Activate virtualenv
- Run pip install -r requirements.txt
- Run in cmd/powershell: ipconfig and find your LAN/Wifi ip address (Dirección IPv4)
- Run python .\server.py -mW 2 -mH 1

```
...
Adaptador de LAN inalámbrica Wi-Fi:

   Sufijo DNS específico para la conexión. . :
   Vínculo: dirección IPv6 local. . . : fe80::511b:3827:d6c8:2228%7
   Dirección IPv4. . . . . . . . . . . . . . : 192.168.0.21
   Máscara de subred . . . . . . . . . . . . : 255.255.255.0
   Puerta de enlace predeterminada . . . . . : 192.168.0.1
 ...
```
## Run Client

Run client detection on camera feed:

```sh
python client.py -s server_ip
```
Run client detection on video feed:

```sh
python app.py -v video_path -s server_ip
```

To run and record to AWS S3 put the "-r 1" argument, as shown below:

```sh
python client.py -v video_path -r 1 -s server_ip
```

or

```sh
python client.py -r 1 -s server_ip
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

 - Improve face recognition models accuracy.
 - Try YOLO deeplearning framework for pedestrian recognition.
 - Separete logic into Client, Server and MicroServices.
 
----

License MIT

