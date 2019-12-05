#from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
from SpectralSubSampler import SpectralSubSampler


class SpectralSuperCNN:
    def __init__(self,  hypParams):
        self.hypParams = hypParams
        self.vars = []
        self.conv_weights = []
        self.MSSignal = []

    def __call__(self, x):
        numMSBands = self.hypParams['N']
        numHSBands = self.hypParams['M']
        numFilters = self.hypParams['F']
        sizeFilters = self.hypParams['K']
        numLayers = self.hypParams['L']
        batchSize = self.hypParams['batchSize']

        mu_min = self.hypParams['mu_min']
        mu_max = self.hypParams['mu_max']
        fwhm_min = self.hypParams['fwhm_min']
        fwhm_max = self.hypParams['fwhm_max']

        listOfLambda = self.hypParams['lambda']
        ####TO REMOVE
        print(listOfLambda)

        conv_weights = []  # weights of convolution operation
        sumsampling_weights = []  # all weights from sub-sampling layer
        superres_weights = []  # all weights from superres n/w

        def conv1d(x, ksize=3, nfeats=64, stride=1):
            ndims = x.get_shape().as_list()[-1]
            W = tf.get_variable("weights", [ksize, ndims, nfeats],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable("biases", [nfeats],
                                initializer=tf.constant_initializer(0))
            self.vars.append(W)
            self.vars.append(b)
            self.conv_weights.append(W)
            x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
            return tf.nn.bias_add(x, b)

        def PReLU(x):
            nfeats = x.get_shape().as_list()[-1]
            a = tf.get_variable("prelu_params", [nfeats],
                                initializer=tf.constant_initializer(0))
            pos = tf.nn.relu(x)
            neg = a * (x-tf.abs(x)) * 0.5
            self.vars.append(a)
            return pos + neg

        def fully_connected(x, nfeats=64):
            ndims = x.get_shape().as_list()[-1]
            W = tf.get_variable("weights", [ndims, nfeats],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable("biases", [nfeats],
                                initializer=tf.constant_initializer(0))
            self.vars.append(W)
            self.vars.append(b)
            return tf.add(tf.matmul(x, W), b)

        def res_block(x, ksize=3, nfeats=64, stride=1):
            with tf.variable_scope('first_conv'):
                x1 = conv1d(x, ksize, nfeats, stride)
            with tf.variable_scope('non_linearity'):
                x2 = PReLU(x1)
            with tf.variable_scope('second_conv'):
                x3 = conv1d(x2, ksize, nfeats, stride)
            return x + x3

        ###################################################################
        # definition of model starts

        with tf.variable_scope('spectral_subsampler'):
            self.subsampler_layer = SpectralSubSampler(numMSBands=numMSBands, listOfLambda=listOfLambda,
                                                       listOfCenterMin=mu_min, listOfCenterMax=mu_max, listOfFWHMMin=fwhm_min, listOfFWHMMax=fwhm_max)
            x1 = self.subsampler_layer(x)
        self.MSSignal = x1    
             
        with tf.variable_scope('upsample_layer'):
            x2 = PReLU(fully_connected(x1, nfeats=numHSBands))
            x2 = tf.reshape(x2, shape=(-1, numHSBands, 1))

        with tf.variable_scope('first_layer'):
            x3 = PReLU(conv1d(x=x2, ksize=sizeFilters, nfeats=numFilters))

        x_mov = x3
        for i in range(numLayers):
            with tf.variable_scope('internal_layer%d' % i):
                x_mov = res_block(x=x_mov, ksize=sizeFilters,
                                  nfeats=numFilters)
        x_mov = tf.add(x_mov, x3)

        with tf.variable_scope('last_layer'):
            x4 = PReLU(conv1d(x=x_mov, ksize=sizeFilters, nfeats=1))

        return tf.reshape(x4,  shape=(-1, numHSBands))

    def get_conv_weights(self):
        return self.conv_weights

    def get_supnetwork_weights(self):
        return self.vars

    def get_subsampler_weights(self):
        return self.subsampler_layer.get_parameters()

    def get_bands(self):
        return self.subsampler_layer.get_bands()
    
    def get_MSSignal(self):
        return self.MSSignal
