"""
Reference: 

official yolov3 implementation  https://github.com/pjreddie/darknet
Ayoosh Kathuria https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
YunYang1994  https://github.com/YunYang1994/tensorflow-yolov3
"""


import numpy as np
from core.yolov3 import build
from core.utils import configManager


def loadWeights(model,cfgfile,weightfile):
    fp = open(weightfile, "rb")

    np.fromfile(fp, dtype=np.int32, count=5)
    blocks = configManager(cfgfile)
    for i, block in enumerate(blocks[1:]):

        if (block["type"] == "convolutional"):
            conv_layer = model.get_layer('conv_' + str(i))
            print("layer: ",i+1,conv_layer)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]
            if "batch_normalize" in block:

                norm_layer = model.get_layer('bnorm_' + str(i))
                print("layer: ",i+1,norm_layer)
                size = np.prod(norm_layer.get_weights()[0].shape)

                bn_weights = np.fromfile(fp, dtype=np.float32, count=4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)

            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(fp, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if "batch_normalize" in block:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

    assert len(fp.read()) == 0, 'transform weight failed'
    fp.close()


def main():

    #weightfile = "weights/yolov3.weights"
    #cfgfile = "cfg/yolov3.cfg"
    #num_classes = 80

    weightfile = "weights/yolov2-voc.weights"
    cfgfile = "cfg/yolov2-voc.cfg"
    num_classes = 19

    #weightfile = "weights/extraction.conv.weights"
    #cfgfile = "cfg/extraction.conv.cfg"
    #num_classes = 80

    model_size = (416, 416, 3)

    model=build(cfgfile,model_size,num_classes)
    loadWeights(model,cfgfile,weightfile)

    try:
        model.save_weights('weights/yolov2-voc.weights.tf')
        print('\nThe file \'yolov2-voc.weights.tf\' has been saved successfully.')
    except IOError:
        print("Couldn't write the file \'yolov2_voc.weights\'.")


if __name__ == '__main__':
    main()






