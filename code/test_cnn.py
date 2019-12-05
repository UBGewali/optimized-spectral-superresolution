import numpy as np
import tensorflow as tf
import yaml
from SpectralSuperCNN import SpectralSuperCNN
import pandas
import os
import tqdm


def read_image(path):
    return np.load(path)['x']

def write_image(path, x):
    np.savez(path, x=x)

def test_model():
    rootDatasetDir = './dataset'
    if not os.path.exists(os.path.join(rootDatasetDir,'pred')):    os.mkdir(os.path.join(rootDatasetDir,'pred'))

    hypParams = yaml.load(open("./best_model/hypparams.yaml", "r"))

    numHSBands = hypParams['M']
    numMSBands = hypParams['N']
    batchSize = hypParams['batchSize']    
    batchSize = 2048 * 16


    h = tf.placeholder(tf.float32, [None,numHSBands])
    model = SpectralSuperCNN(hypParams)
    hpred = model(h)
    mssignal = model.get_MSSignal()
    bandCenters, bandFWHMs = model.get_bands()

    # Configuring testing
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
 

    with tf.Session(config=config) as sess:  
        sess.run(init)
        saver.restore(sess, './best_model/best_model.ckpt')

        for f in tqdm.tqdm(os.listdir(os.path.join(rootDatasetDir,'test'))):
            img = read_image(os.path.join(rootDatasetDir, 'test', f))
            [nrows,ncols,nbands] = img.shape
            spectra = img.reshape((-1,nbands))

            predHSI = []
            predMSI = []
            for j in range(0,spectra.shape[0]-batchSize+1, batchSize):
                x = spectra[j:(j+batchSize),:].reshape((batchSize,numHSBands))
                y,msi = sess.run([hpred, mssignal], feed_dict={h:x})
                predHSI.append(y)
                predMSI.append(msi)
            x = spectra[(-batchSize):,:].reshape((batchSize,numHSBands))
            y,msi = sess.run([hpred, mssignal], feed_dict={h:x})
            filler_num = spectra.shape[0]-(j+batchSize)
            if filler_num > 0: 
                predHSI.append(y[-filler_num:,:])
                predMSI.append(msi[-filler_num:,:])
            predHSI = np.vstack(predHSI)
            predMSI = np.vstack(predMSI)    
            predHSI = predHSI.reshape([nrows,ncols,nbands])
            predMSI = predMSI.reshape([nrows,ncols,predMSI.shape[-1]])
        
            write_image(os.path.join(rootDatasetDir, 'pred', f[:-4]+'_MSI.npz'), predMSI)        
            write_image(os.path.join(rootDatasetDir, 'pred', f), predHSI)        
        

if __name__ == '__main__':
    print("Testing Model.......")
    test_model()
