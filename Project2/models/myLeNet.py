"""
LeNet model for Keras.

Created on Sunday April 15 10:27:10 2018

@author: Sainan Liu

# Reference
- https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
- http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
"""
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Flatten, Dense, Dropout
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape

def LeNet(include_top = True, input_shape= (28, 28, 1), num_classes = 10):
    """ Initiate the LeNet architecture.
    # Arguments
    include_top: whether to include the 2 fully-connected layers at the top of the network.
    input_shape: optional shape tuple, only to be specified if 'inlucde_top' is False (otherwise the input shape has to be (28, 28, 1) with 'tf' dim ordering).
    num_classes: optional number of classes to classify images into, only to be specified if 'inlucde_tope' is True.
    """
    input_shape = _obtain_input_shape(input_shape,
                                      default_size = 28,
                                      min_size = 28,
                                      data_format = K.image_data_format(),
                                      require_flatten=include_top)
    # initialize the model input
    img_input = Input(shape=input_shape)

    # first set of CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding = "same",
               activation = 'relu', name = 'block1_conv1')(img_input)

    x = MaxPooling2D(pool_size=(2, 2),
                     name='block1_pool')(x)

    # second set of CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding = "same",
               activation='relu', name = 'block2_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2),
                     name='block2_pool')(x)

    x = Dropout(0.25)(x)

    if include_top:
        # set of FC => RELU layers
        x = Flatten(name='flatten')(x)
        x = Dense(128, activation='relu', name='fc1')(x)

        # softmax classifier
        x = Dense(num_classes, activation='softmax', name='predications')(x)

    # create model.
    model = Model(img_input, x, name='LeNet')
    # return the constructed network architecture
    return model
