# Inception-v3, model from the paper:
# "Rethinking the Inception Architecture for Computer Vision"
# http://arxiv.org/abs/1512.00567
# Original source:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
# License: http://www.apache.org/licenses/LICENSE-2.0

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/inception_v3.pkl

import theano
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import softmax


def preprocess(image, normalization=False, rollaxis=False):
    # Expected input: RGB uint8 image
    # Input to network should be bc01, 299x299 pixels, scaled to [-1, 1].
    import skimage.transform
    import numpy as np

    im = skimage.transform.resize(image, (3, 299, 299), preserve_range=True)
    if normalization:
        im = (im - 128) / 128.
    if rollaxis:
        im = np.rollaxis(im, 2)[np.newaxis].astype('float32')
    return im


def bn_conv(input_layer, **kwargs):
    l = Conv2DLayer(input_layer, **kwargs)
    l = batch_norm(l, epsilon=0.001)
    return l


def inceptionA(input_layer, nfilt):
    # Corresponds to a modified version of figure 5 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionB(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionC(input_layer, nfilt):
    # Corresponds to figure 6 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
    l3 = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionD(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    l1 = bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
    l2 = bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionE(input_layer, nfilt, pool_mode):
    # Corresponds to figure 7 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2a = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
    l2b = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3a = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
    l3b = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])


def build_network(pooling='avg'):
    net = {}

    net['input'] = InputLayer((None, 3, 299, 299))
    net['conv'] = bn_conv(net['input'],
                          num_filters=32, filter_size=3, stride=2)
    net['conv_1'] = bn_conv(net['conv'], num_filters=32, filter_size=3)
    net['conv_2'] = bn_conv(net['conv_1'],
                            num_filters=64, filter_size=3, pad=1)
    net['pool'] = Pool2DLayer(net['conv_2'], pool_size=3, stride=2, mode='max')

    net['conv_3'] = bn_conv(net['pool'], num_filters=80, filter_size=1)

    net['conv_4'] = bn_conv(net['conv_3'], num_filters=192, filter_size=3)

    net['pool_1'] = Pool2DLayer(net['conv_4'],
                                pool_size=3, stride=2, mode='max')
    net['mixed/join'] = inceptionA(
        net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
    net['mixed_1/join'] = inceptionA(
        net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_2/join'] = inceptionA(
        net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_3/join'] = inceptionB(
        net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

    net['mixed_4/join'] = inceptionC(
        net['mixed_3/join'],
        nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

    net['mixed_5/join'] = inceptionC(
        net['mixed_4/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_6/join'] = inceptionC(
        net['mixed_5/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_7/join'] = inceptionC(
        net['mixed_6/join'],
        nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

    net['mixed_8/join'] = inceptionD(
        net['mixed_7/join'],
        nfilt=((192, 320), (192, 192, 192, 192)))

    net['mixed_9/join'] = inceptionE(
        net['mixed_8/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode='average_exc_pad')

    net['mixed_10/join'] = inceptionE(
        net['mixed_9/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode='max')

    if pooling == "max":
        net['pool3'] = GlobalPoolLayer(net['mixed_10/join'], pool_function=theano.tensor.max)
    elif pooling in ("avg","mean"):
        net['pool3'] = GlobalPoolLayer(net['mixed_10/join'], pool_function=theano.tensor.mean)
    else:
        raise "pooling not specified"

    net['softmax'] = DenseLayer(
        net['pool3'], num_units=1008, nonlinearity=softmax)

    return net

import os
import config
from inception_v3 import build_network, preprocess
from sysfile import exists_or_download_list
from sysfile import create_dir
from sysfile import unpickle

#pretrained weights
IV3_MODEL = [
    ("1xcx7s8EtEKfb5ScLpNIcXV2Qe6KMCqLb", False, 'google'),
    ("s3://lasagne/recipes/pretrained/imagenet/inception_v3.pkl", False, 'amazon')
]

# Load and wrap the Inception model
def load_and_build_inception_net(pooling="avg"):
    create_dir(config.SAVE)
    inception_model = build_network(pooling)
    model_path = os.path.join(config.SAVE, "inception_v3.pkl")
    exists_or_download_list(model_path, IV3_MODEL)
    pretrained_weights = unpickle(model_path)
    lasagne.layers.set_all_param_values(inception_model['softmax'], pretrained_weights['param values'])
    return inception_model
