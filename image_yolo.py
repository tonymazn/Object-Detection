"""
Refernece: 
official yolov3 implementation  https://github.com/pjreddie/darknet
Ayoosh Kathuria https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
YunYang1994  https://github.com/YunYang1994/tensorflow-yolov3
"""


import tensorflow as tf
from core.utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from core.yolov3 import build

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

sizeOfModel = (416, 416,3)
numberOfClass = 80
className = './data/coco.names'
maxOutputSize = 40
maxOutputSizePerClass= 20
iouThreshold = 0.5
confidenceThreshold = 0.6

cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'
imgFilename = "data/images/test.jpg"

def main():

    model = build(cfgfile,sizeOfModel,numberOfClass)
    model.load_weights(weightfile)

    class_names = load_class_names(className)

    image = cv2.imread(imgFilename)
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    resized_frame = resize_image(image, (sizeOfModel[0],sizeOfModel[1]))
    pred = model.predict(resized_frame)

    boxes, scores, classes, nums = output_boxes( \
        pred, sizeOfModel,
        max_output_size=maxOutputSize,
        max_output_size_per_class=maxOutputSizePerClass,
        iou_threshold=iouThreshold,
        confidence_threshold=confidenceThreshold)

    aclass = classes[0]
    image = np.squeeze(image)
    img = draw_outputs(image, boxes, scores, classes, nums, class_names)

    win_name = 'Image detection'
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
