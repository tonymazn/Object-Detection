"""
Reference: 

YunYang1994  https://github.com/YunYang1994/tensorflow-yolov3/blob/master/image_demo.py
"""



import tensorflow as tf
import cv2
import numpy as np
from core.yolov3 import build
from core.utils import getClassNames, getOutputBoxes, drawOutputs, resizeImage

win_name = 'Image object detection'
className = './data/coco.names'
cfgfile = './cfg/yolov3.cfg'
weightfile = './weights/yolov3_weights.tf'
imgFilename = "./data/images/demo.jpg"

sizeOfModel = (416, 416,3)
numberOfClass = 80
maxOutputSize = 40
maxOutputSizePerClass = 20
iouThreshold = 0.5
confidenceThreshold = 0.4

def main():
    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)
    model = build(cfgfile,sizeOfModel,numberOfClass)
    model.load_weights(weightfile)
    classNames = getClassNames(className)
    image = cv2.imread(imgFilename)
    image = np.array(image)
    image = tf.expand_dims(image, 0)
    resized_frame = resizeImage(image, (sizeOfModel[0],sizeOfModel[1]))
    pred = model.predict(resized_frame)
    boxes, scores, classes, nums = getOutputBoxes(pred, 
        sizeOfModel, 
        maxOutputSize=maxOutputSize,
        maxOutputSizePerClass=maxOutputSizePerClass,
        iouThreshold=iouThreshold,
        confidenceThreshold=confidenceThreshold)
    aclass = classes[0]
    image = np.squeeze(image)
    img = drawOutputs(image, boxes, scores, classes, nums, classNames)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
