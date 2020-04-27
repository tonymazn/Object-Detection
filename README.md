# YOLOv3 Tensorflow

Here's the image output example:

![Detection Example](Image_detection.png)

Here's the video output example:
https://youtu.be/7O721Zt6wE8

Here's the camera output example:
https://youtu.be/GHK44tlMlcA

-
This project is a Python base tool to demonstrate Object Detecion by YOLO v3.

Installation
-
Getting the tool to work is simple, download/clone the package, unzip the package and go to the ObjectDetection-master folder and then run the following command:

```
step 1
Create Virtual Environment
    conda create -n yolov3 python=3.7

    conda activate yolov3

step 2
Install the requirements packages
(yolov3)$ pip install -r requirement.txt

you are ready to go
```

System Structure
-



    .
    ├─ .vs                                       # Visual Studio files
    ├─ cfg
    │     ├── yolov2-voc.cfg                     # YOLO2 VOC  dataset config file
    │     └── yolov3.cfg                         # YOLO3 COCO dataset config file
    ├── core
    │     ├── utils.py                           # utils
    │     └── yolov3.py                          # YOLOv3 tensorflow model file
    ├── data 
    │     ├───── images                          # imput image files
    │     │         ├── demo.jpg               
    │     │         ├── demo1.jpg              
    │     │         ├── demo2.jpg              
    │     │         └── dog.jpg 
    │     ├── coco.names                         # YOLO3 coco dataset class file 
    │     └── voc.names                          # YOLO3 VOC dataset class file 
    ├── weights
    │     ├── yolov2-voc.weights                 # YOLO VOC dataset pre-train weight file                       
    │     ├── yolov2-voc.weights.tf.data-00000-of-00002
    │     ├── yolov2-voc.weights.tf.data-00001-of-00002
    │     ├── yolov2-voc.weights.tf.index        # Tensorflow format YOLO VOC weight file
    │     ├── yolov3.weights                     # YOLO COCO dataset pre-train weight file                        
    │     ├── yolov3_weights.tf.data-00000-of-00002
    │     ├── yolov3_weights.tf.data-00001-of-00002
    │     ├── yolov3_weights.tf.index            # Tensorflow format YOLO COCO weight file
    │     └── note.txt                           

    ├─ README.md
	├─ camera_yolo.py                            # Object Detaction by camera
	├─ video_yolo.py                             # Object Detaction by video file
	├─ image_yolo.py                             # Object Detaction by image file
	├─ transform_weights.py                      # Transform YOLO weight file to tensorform format
	├─ requirements.txt                          # Install requirement list
    ├─ SECURITY.md                               # github SECURITY file
    └─ LICENSE                                   # License 
