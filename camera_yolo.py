"""
Reference: 

official yolov3 implementation  https://github.com/pjreddie/darknet
Ayoosh Kathuria https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
YunYang1994  https://github.com/YunYang1994/tensorflow-yolov3
jaskarannagi19 https://github.com/jaskarannagi19/yolov3
"""


import tensorflow as tf

from core.utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import time

from core.yolov3tensorflow import build

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


modelSize = (416, 416,3)
numOfClasses = 80
className = './data/coco.names'
maxOutputSize = 100
maxOutputSizePerClass= 20
iouThreshold = 0.5
confidenceThreshold = 0.5


cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'

def main():

    model = build(cfgfile,modelSize,numOfClasses)

    model.load_weights(weightfile)

    class_names = load_class_names(className)



    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)

    cap = cv2.VideoCapture(0)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (modelSize[0],modelSize[1]))

            pred = model.predict(resized_frame)

            boxes, scores, classes, nums = output_boxes( \
                pred, modelSize,
                max_output_size=maxOutputSize,
                max_output_size_per_class=maxOutputSizePerClass,
                iou_threshold=iouThreshold,
                confidence_threshold=confidenceThreshold)

            img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
            cv2.imshow(win_name, img)

            stop = time.time()

            seconds = stop - start

            fps = 1 / seconds
            print("Estimated frames per second : {0}".format(fps))

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('Detections have been performed successfully.')


if __name__ == '__main__':
    main()
