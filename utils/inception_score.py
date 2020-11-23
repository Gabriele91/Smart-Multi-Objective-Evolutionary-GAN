import os
import lasagne
import config
from inception_v3 import preprocess, load_and_build_inception_net
from sysfile import exists_or_download_list
from sysfile import create_dir
from sysfile import unpickle

import theano
from math import floor
import numpy as np
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy import asarray
from numpy.random import shuffle
from skimage.transform import resize

# scale an array of images to a new size
def scale_images(images, new_shape):
    imgs_shape = (images.shape[0], *new_shape)
    images_list = np.zeros(shape=imgs_shape, dtype=theano.config.floatX)
    for i, image in enumerate(images):
        images_list[i] = resize(image, new_shape, preserve_range=True)
    return images_list

#model predict
def model_predict(model, image):
    lasagne_eval = lasagne.layers.get_output(model['softmax'], image, deterministic=True).eval()
    return np.array(lasagne_eval)

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, model, n_split=10, eps=1E-16):
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        # pre-process images, scale to [-1,1]
        # convert from uint8 to float32
        # scale images to the required size
        subset = scale_images(subset, (3, 299, 299))
        # predict p(y|x)
        p_yx = model_predict(model,subset)
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

#compute score
def inception_score(images, model = None, n_split=100):
    if model is None:
        model = load_and_build_inception_net(pooling='avg')
    return calculate_inception_score(images, model, n_split)

class InceptionScore(object):

    def __init__(self,n_split = 100):
        self.model = load_and_build_inception_net(pooling='avg')
        self.n_split = n_split

    def __call__(self,images):
        return calculate_inception_score(images, self.model, self.n_split)

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataset.datasets import cifar10
    db = cifar10()
    X = db["X_train"]
    print(inception_score(X, n_split=100))
