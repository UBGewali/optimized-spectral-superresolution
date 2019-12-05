#from __future__ import print_function
import tensorflow as tf
import numpy as np

def MSE(x1,x2):
    return tf.reduce_mean(tf.square(x1-x2))

def SAM(x1,x2):
    x1x2 = tf.reduce_sum(tf.multiply(x1,x2),axis=1)
    x1x1 = tf.reduce_sum(tf.multiply(x1,x1),axis=1)
    x2x2 = tf.reduce_sum(tf.multiply(x2,x2),axis=1)
    z = tf.acos(x1x2 / (tf.sqrt(tf.multiply(x1x1,x2x2))+1e-12) )
    return tf.reduce_mean(z)
    
def first_derivative(x):
    ndims = x.get_shape().as_list() + [1] 
    h = tf.constant([1./12., -2./3., 0., 2./3., -1./12.]) #first derivative filter
    h = tf.reshape(h,(5,1,1))
    y = tf.nn.conv1d(tf.reshape(x,shape=ndims),h,stride=1,padding="SAME")
    return y

def second_derivative(x):
    ndims = x.get_shape().as_list() + [1] 
    h = tf.constant([-1./12., 4./3., -5./2., 4./3., -1./12.]) #second derivative filter
    h = tf.reshape(h,(5,1,1))
    y = tf.nn.conv1d(tf.reshape(x,shape=ndims),h,stride=1,padding="SAME")
    return y

def spectral_cost(x1, x2, w1=0.5, w2=0.5):
    x1_der1 = first_derivative(x1)
    x2_der1 = first_derivative(x2)
    
    x1_der2 = second_derivative(x1)
    x2_der2 = second_derivative(x2)

    dist = MSE(x1,x2) + w1 * MSE(x1_der1, x2_der1) + w2 * MSE(x1_der2, x2_der2)
    return dist
