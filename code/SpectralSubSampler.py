#from __future__ import print_function
import tensorflow as tf
import numpy as np   
            
class SpectralSubSampler:
    def __init__(self, numMSBands, listOfLambda, listOfCenterMin=None, listOfCenterMax=None, listOfFWHMMin=None, listOfFWHMMax=None):
        self.numMSBands = numMSBands
        self.numHSBands = len(listOfLambda)
        self.listOfLambda = listOfLambda
        self.listOfCenterMin = listOfCenterMin
        self.listOfCenterMax = listOfCenterMax
        self.listOfFWHMMin = listOfFWHMMin
        self.listOfFWHMMax = listOfFWHMMax

    def __call__(self, x):    
        initValueForSigmoid=4
        defaultMSMinFWHM = 5 * (np.max(self.listOfLambda)-np.min(self.listOfLambda))/len(self.listOfLambda)
        b = 3 #guard gap parameter 

        lambdaSteps = tf.constant(np.array(self.listOfLambda, dtype=np.float32).reshape((self.numHSBands,1)))
        if self.listOfCenterMin is not None:
            centerMin = tf.constant(np.array(self.listOfCenterMin, dtype=np.float32).reshape((1,self.numMSBands)))
        else:
            centerMin = tf.constant(np.array([np.min(self.listOfLambda)]*self.numMSBands, dtype=np.float32).reshape((1,self.numMSBands)))
        if self.listOfCenterMax is not None:
            centerMax = tf.constant(np.array(self.listOfCenterMax, dtype=np.float32).reshape((1,self.numMSBands)))
        else:
            centerMax = tf.constant(np.array([np.max(self.listOfLambda)]*self.numMSBands, dtype=np.float32).reshape((1,self.numMSBands)))
        if self.listOfFWHMMin is not None:
            FWHMMin = tf.constant(np.array(self.listOfFWHMMin, dtype=np.float32).reshape((1,self.numMSBands)))
        else:
            FWHMMin = tf.constant(np.array([defaultMSMinFWHM]*self.numMSBands, dtype=np.float32).reshape((1,self.numMSBands)))
        if self.listOfFWHMMax is not None:
            FWHMMax = tf.constant(np.array(self.listOfFWHMMax, dtype=np.float32).reshape((1,self.numMSBands)))


        mu0 = tf.get_variable("mu0", shape=[1,self.numMSBands],initializer=tf.random_uniform_initializer(minval=-initValueForSigmoid,maxval=initValueForSigmoid))
        
        if self.listOfFWHMMax is not None:
            sigma0 = tf.get_variable("sigma0", shape=[1,self.numMSBands],initializer=tf.random_uniform_initializer(minval=-initValueForSigmoid,maxval=initValueForSigmoid))
        else:
            sigma0 = tf.get_variable("sigma0", shape=[1,self.numMSBands],initializer=tf.random_uniform_initializer(minval=0,maxval=(np.max(self.listOfLambda)-np.min(self.listOfLambda))/6))

        if self.listOfFWHMMax is not None:
            sigma = FWHMMin/2.355+tf.nn.sigmoid(sigma0)*(FWHMMax/2.355-FWHMMin/2.355)
        else:
            sigma = FWHMMin/2.355+tf.abs(sigma0)
        
        mu = (centerMin+b*sigma)+tf.nn.sigmoid(mu0)*(centerMax-centerMin-2*b*sigma)
             
        z_2 = tf.div(tf.square(lambdaSteps-mu),2*tf.square(sigma)) 
        Z = tf.sqrt(2*np.pi*tf.square(sigma))
        Kmulti = tf.exp(-z_2) / Z
        
        y = tf.matmul(x, Kmulti) 

        self.mu0 = mu0
        self.sigma0 = sigma0
        self.mu = mu
        self.sigma = sigma
        return y
    
    def get_parameters(self):
        return [self.mu0, self.sigma0]
 
    def get_bands(self):
        return [self.mu, 2.355*self.sigma]

