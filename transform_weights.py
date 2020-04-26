"""
Reference: 

official yolov3 implementation  https://github.com/pjreddie/darknet
Ayoosh Kathuria https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
YunYang1994  https://github.com/YunYang1994/tensorflow-yolov3
jaskarannagi19 https://github.com/jaskarannagi19/yolov3
"""

import numpy as np
from core.yolov3tensorflow import YOLOv3Net
from core.yolov3tensorflow import parse_cfg




def load_weights(model,cfgfile,weightfile):
    # Open the weights file
    fp = open(weightfile, "rb")

    # Skip 5 header values
    np.fromfile(fp, dtype=np.int32, count=5)

    # The rest of the values are the weights
    blocks = parse_cfg(cfgfile)

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
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                fp, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if "batch_normalize" in block:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

    assert len(fp.read()) == 0, 'failed to read all data'
    fp.close()


def main():

#    weightfile = "weights/yolov3.weights"
#    cfgfile = "cfg/yolov3.cfg"

    weightfile = "weights/yolov2-voc.weights"
    cfgfile = "cfg/yolov2-voc.cfg"

    weightfile = "weights/extraction.conv.weights"
    cfgfile = "cfg/extraction.conv.cfg"

    model_size = (416, 416, 3)
    num_classes = 80

    model=YOLOv3Net(cfgfile,model_size,num_classes)
    load_weights(model,cfgfile,weightfile)

    try:
        model.save_weights('weights/extraction_conv_weights.tf')
        print('\nThe file \'yolov3_weights.tf\' has been saved successfully.')
    except IOError:
        print("Couldn't write the file \'yolov2_voc.weights\'.")


if __name__ == '__main__':
    main()






