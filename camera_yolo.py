"""
Reference: 

official yolov3 implementation  https://github.com/pjreddie/darknet
Ayoosh Kathuria https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
YunYang1994  https://github.com/YunYang1994/tensorflow-yolov3
jaskarannagi19 https://github.com/jaskarannagi19/yolov3
"""


import tensorflow as tf
import cv2
import time
from core.yolov3 import build
from core.utils import getClassNames, getOutputBoxes, drawOutputs, resizeImage, statistics, startStatistics, isExisting

win_name = 'Yolov3 object detection by camera'
className = './data/coco.names'
cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'

modelSize = (416, 416,3)
numOfClasses = 80
maxOutputSize = 100
maxOutputSizePerClass = 20
iouThreshold = 0.5
confidenceThreshold = 0.5

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
def main():
    model = build(cfgfile,modelSize,numOfClasses)
    model.load_weights(weightfile)
    class_names = getClassNames(className)
    cv2.namedWindow(win_name)
    cap = cv2.VideoCapture(0)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        while True:
            startStatistics()
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resizeImage(resized_frame, (modelSize[0],modelSize[1]))
            pred = model.predict(resized_frame)
            boxes, scores, classes, nums = getOutputBoxes(pred, 
                modelSize,
                maxOutputSize=maxOutputSize,
                maxOutputSizePerClass=maxOutputSizePerClass,
                iouThreshold=iouThreshold,
                confidenceThreshold=confidenceThreshold)
            img = drawOutputs(frame, boxes, scores, classes, nums, class_names)
            cv2.imshow(win_name, img)
            statistics()
            if isExisting():
                break
    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('The job has been done successfully.')


if __name__ == '__main__':
    main()
