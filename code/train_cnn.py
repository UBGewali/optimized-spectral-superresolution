import numpy as np
import tensorflow as tf
import yaml
from SpectralSuperCNN import SpectralSuperCNN
import SpectralLoss
from SpectraLoader import SpectraLoader
import pandas
import os

def train_model():
    if not os.path.exists('./model'):    os.mkdir('./model')
    
    hypParams = yaml.load(open("hypparams.yaml", "r"), Loader=yaml.FullLoader)

    numHSBands = hypParams['M']
    numMSBands = hypParams['N']
    batchSize = hypParams['batchSize']    
    lrSubSampler = hypParams['lr1']    
    lrSupNetwork = hypParams['lr2']    
    weightDecayFactor = hypParams['weightDecayFactor']    
    w1 = hypParams['w1']
    w2 = hypParams['w2']

    valFrequency = hypParams['valFrequency']
    num_val_batches = hypParams['num_val_batches']
    patienceIterations = hypParams['patienceIterations']


    h = tf.placeholder(tf.float32, [batchSize,numHSBands])
    model = SpectralSuperCNN(hypParams)
    hpred = model(h)
    bandCenters, bandFWHMs = model.get_bands()

    regularizer =  tf.reduce_mean(tf.stack([tf.nn.l2_loss(w) for w in model.get_conv_weights()]))
    cost = SpectralLoss.spectral_cost(h,hpred, w1, w2) + weightDecayFactor * regularizer
    mse = SpectralLoss.MSE(h,hpred) 
    sam = SpectralLoss.SAM(h,hpred) 


    optSubSampler = tf.train.AdamOptimizer(learning_rate=lrSubSampler).minimize(cost, var_list=model.get_subsampler_weights()) 
    optSupNet = tf.train.AdamOptimizer(learning_rate=lrSupNetwork).minimize(cost, var_list=model.get_supnetwork_weights()) 
    optimizer = tf.group(optSubSampler, optSupNet) 
  
    # Configuring training
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
 
    bestVal = np.inf
    bestBands = {'centers': [], 'FWHMs': []}
    itersWithoutImprov = 0
    nBatches = int(1e8)
    infoFile = './model/training.csv'

    trainGenerator = SpectraLoader("train")
    valGenerator = SpectraLoader("validation")

    listOfMetrics = []

    with tf.Session(config=config) as sess:  
        sess.run(init)
        
        for countBatch in range(nBatches):    
            xtrain = trainGenerator.get_batch(batchSize)
            sess.run(optimizer, feed_dict={h:xtrain})
	    
            if countBatch % valFrequency == 0:
                trainLoss = sess.run(cost, feed_dict={h:xtrain})
                trainRMSE = np.sqrt(sess.run(mse, feed_dict={h:xtrain}))

                listOfValSAM = []
                listOfValRMSE = []
                for i in range(num_val_batches):
                    xval = valGenerator.get_batch(batchSize)
                    listOfValSAM.append(sess.run(sam, feed_dict={h:xval}))
                    listOfValRMSE.append(np.sqrt(sess.run(mse, feed_dict={h:xval})))

                valRMSE = np.mean(listOfValRMSE)
                valSAM = np.mean(listOfValSAM)

                if np.isnan(valSAM):    
                    bestVal = np.pi/2 #max SAM value
                    break

                if valSAM < bestVal:
                    [listOfBandCenters, listOfBandFWHMs] = sess.run([bandCenters, bandFWHMs])
                    bestVal = valSAM
                    bestBands['centers'] = listOfBandCenters.ravel().tolist()
                    bestBands['FWHMs'] = listOfBandFWHMs.ravel().tolist()
                    saver.save(sess, './model/best_model.ckpt')        
                    itersWithoutImprov = 0
                else:
                    itersWithoutImprov += 1


                if  itersWithoutImprov > patienceIterations:
                    break

                metricNames = ['Batch#', 'Train loss', 'Train RMSE', 'Validation RMSE', 'Validation SAM'] + ['Band Center %d'%(i+1) for i in range(len(bestBands['centers']))] + ['Band FWHM %d'%(i+1) for i in range(len(bestBands['FWHMs']))]  
                
                metrics = [countBatch, trainLoss, trainRMSE, valRMSE, valSAM] + bestBands['centers'] + bestBands['FWHMs']
                listOfMetrics.append(metrics)

                printMsg = ["%s: %.4f"%x for x in zip(metricNames, metrics)]
                for i in range(len(printMsg)):
                    print(printMsg[i])
                print("--------------------------------------------------------------------")

                metricsTable = pandas.DataFrame(data=listOfMetrics, columns=metricNames)
                metricsTable.to_csv(infoFile)
 
        return bestVal	 
	
        

if __name__ == '__main__':
    print("Training Model.......")
    val = train_model()
    np.savetxt("current_val.csv", np.array([val,]))
