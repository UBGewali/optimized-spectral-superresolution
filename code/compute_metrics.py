import os
import h5py
import numpy as np
import tqdm

def load_image(filename):
    return np.load(filename)['x']

def main():
    datasetPath = './dataset'

    rmseList = []
    nrmseList = []
    perBandNRMSEList = []
    samList = []    
    r2List = []
    for filename in tqdm.tqdm(os.listdir(os.path.join(datasetPath,'test'))): #metric are computer per image for this dataset
        image = load_image(os.path.join(datasetPath,'test',filename))
        [nrows,ncols,nbands] = image.shape
        spectraGT = image.reshape((nrows*ncols,nbands))

        image = load_image(os.path.join(datasetPath,'pred',filename))
        [nrows,ncols,nbands] = image.shape
        spectraPred = image.reshape((nrows*ncols,nbands))

        rmseList.append(RMSE(spectraGT,spectraPred))
        nrmseList.append(NRMSE(spectraGT,spectraPred))
        samList.append(SAM(spectraGT,spectraPred))
        r2List.append(coerr(spectraGT,spectraPred))
        perBandNRMSEList.append(perBandNRMSE(spectraGT,spectraPred))
            
    rmse = np.around(np.mean(rmseList),3)
    nrmse = np.around(np.mean(nrmseList),4)
    sam = np.around(np.mean(samList),4)
    r2 = np.around(np.mean(r2List),3)
    perBandNRMSEall = np.vstack(perBandNRMSEList)
    perBandNRMSE_mean = perBandNRMSEall.mean(axis=0)
    perBandNRMSE_std = perBandNRMSEall.std(axis=0)
    
    
    print('RMSE: %g  ' % rmse)
    print('NRMSE: %g  ' % nrmse)
    print('SAM: %g  ' % sam)
    print('r2: %g  ' % r2)
    print('Per band mean NRMSE:')
    print(perBandNRMSE_mean)
    print('Per band std. dev. NRMSE:')
    print(perBandNRMSE_std)


def RMSE(xtrue,xpred):
    err = np.square((xtrue-xpred)*255./xtrue.max())
    return np.sqrt(err.mean())

def NRMSE(xtrue,xpred):
    err = np.square((xtrue-xpred)/xtrue)
    return np.sqrt(err.mean())

def perBandNRMSE(xtrue,xpred):
    err = np.square((xtrue-xpred)/xtrue)
    return np.sqrt(err.mean(axis=0))

def SAM(x1,x2):
    x1x2 = np.sum(x1*x2,axis=1)
    x1x1 = np.sum(x1*x1,axis=1)
    x2x2 = np.sum(x2*x2,axis=1)
    return np.arccos(x1x2 / (np.sqrt(x1x1*x2x2)+1e-12) ).mean()

def coerr(x,y):
    mx = x.mean(axis=1,keepdims=True)
    my = y.mean(axis=1,keepdims=True)
    
    x = x - mx
    y = y - my
    
    r = np.sum(x*y,axis=1)/np.sqrt(np.sum(x*x,axis=1)*np.sum(y*y,axis=1))
    return r


if __name__ == '__main__':
    main()
