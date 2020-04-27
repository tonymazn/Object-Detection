"""
Reference: 

official yolov3 implementation  https://github.com/pjreddie/darknet
Ayoosh Kathuria https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
YunYang1994  https://github.com/YunYang1994/tensorflow-yolov3
jaskarannagi19 https://github.com/jaskarannagi19/yolov3
"""


import tensorflow as tf
import numpy as np
import cv2

def configManager(cfgfile):
    with open(cfgfile, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    holder = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)
    return blocks

def resizeImage(inputs, modelsize):
    inputs= tf.image.resize(inputs, modelsize)
    return inputs


def getOutputBoxes(inputs,modelSize, maxOutputSize, maxOutputSizePerClass,
                 iouThreshold, confidenceThreshold):

    centerX, centerY, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    topLeftX = centerX - width / 2.0
    topLeftY = centerY - height / 2.0
    bottomRightX = centerX + width / 2.0
    bottomRightY = centerY + height / 2.0

    inputs = tf.concat([topLeftX, topLeftY, bottomRightX,
                        bottomRightY, confidence, classes], axis=-1)

    boxes_dicts = nonMaxSuppression(inputs, modelSize, maxOutputSize, maxOutputSizePerClass, iouThreshold, confidenceThreshold)

    return boxes_dicts

def drawOutputs(img, boxes, objectness, classes, nums, class_names):
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes=np.array(boxes)

    for i in range(nums):
        if class_names[int(classes[i])] != 'aperson':
            x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
            x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))

            img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)

            img = cv2.putText(img, '{} {:.2f}'.format(class_names[int(classes[i])], objectness[i]),
                              (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    return img



def getClassNames(file_name):
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def nonMaxSuppression(inputs, model_size, max_output_size,
                        max_output_size_per_class, iou_threshold, confidence_threshold):
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox=bbox/model_size[0]

    scores = confs * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return boxes, scores, classes, valid_detections

def quit():
    return (cv2.waitKey(1) & 0xFF) == ord('c')
           



