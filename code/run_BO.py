import numpy as np
import GPyOpt        
import pandas
import yaml
import os
import shutil
import glob
import os

class objective:
    def __init__(self, HSBands, numMSBands, minMSCenters, maxMSCenters, minMSFWHMs, maxMSFWHMs,\
                 batchSize, valFrequency, num_val_batches, patience):
        self.current_best = np.inf
        self.listOfParams = []

        self.csvHistory = './best_model/BO_history.csv'
        
        self.numHSBands = len(HSBands)
        self.HSBands = HSBands
        self.numMSBands = numMSBands
        self.minMSCenters = minMSCenters
        self.maxMSCenters = maxMSCenters
        self.minMSFWHMs = minMSFWHMs
        self.maxMSFWHMs = maxMSFWHMs

        self.batchSize = batchSize
        self.valFrequency = valFrequency
        self.num_val_batches = num_val_batches
        self.patience = patience

 
    def __call__(self,x):
        fs = np.zeros((x.shape[0],1))
        for i in range(x.shape[0]):
            num_layers = x[i,0].astype(int).tolist()
            num_features = x[i,1].astype(int).tolist()
            filter_size = x[i,2].astype(int).tolist()
            learning_rate1 = 10.**(x[i,3].astype(np.float32).tolist())
            weight_decay_factor = 10.**(x[i,4].astype(np.float32).tolist())
            learning_rate2 = 10.**(x[i,5].astype(np.float32).tolist())
            w1 = x[i,6].astype(np.float32).tolist()
            w2 = x[i,7].astype(np.float32).tolist()
         
            hypParamsNames = ['M', 'N', 'lambda', 'F', 'K', 'L', 'lr1', 'lr2',\
                             'weightDecayFactor', 'w1', 'w2', 'mu_min', 'mu_max',\
                             'fwhm_min', 'fwhm_max', 'batchSize', 'valFrequency', \
                             'num_val_batches', 'patienceIterations']
            hypParamsValues = [self.numHSBands, self.numMSBands, self.HSBands, num_features, filter_size, num_layers, learning_rate1, learning_rate2,\
                              weight_decay_factor, w1, w2, self.minMSCenters, self.maxMSCenters,\
                              self.minMSFWHMs, self.maxMSFWHMs, self.batchSize, self.valFrequency,\
                              self.num_val_batches, self.patience]

            hypParams = dict(zip(hypParamsNames,hypParamsValues))
            yamlFile = open("hypparams.yaml", "w")
            yaml.dump(hypParams, yamlFile)
            yamlFile.close()

            command = 'python train_cnn.py'
            os.system(command)
            val_perf = float(np.loadtxt('current_val.csv'))

            fs[i] = val_perf
            if val_perf < self.current_best:
                self.current_best = val_perf
                for mfile in glob.glob(r'./model/best_model*'):
                    shutil.copy2(mfile, './best_model')
                shutil.copy2('./model/training.csv', './best_model')
                shutil.copy2('./hypparams.yaml', './best_model')
  
            paramsNames = ['Validation loss',] + hypParamsNames
            paramsValues = [val_perf,] + hypParamsValues
            trainTable = pandas.read_csv('./best_model/training.csv', header=0, index_col=0)
            for i in range(self.numMSBands):
                paramsNames.append('Best Band Center %d'%(i+1))
                paramsValues.append(trainTable.at[trainTable.shape[0]-1,'Band Center %d'%(i+1)])
                paramsNames.append('Best Band FWHM %d'%(i+1))
                paramsValues.append(trainTable.at[trainTable.shape[0]-1,'Band FWHM %d'%(i+1)])

            self.listOfParams.append(paramsValues)
            paraTable = pandas.DataFrame(data=self.listOfParams, columns=paramsNames)
            paraTable.to_csv(self.csvHistory)

        return fs


def main(): 
    if not os.path.exists('./best_model'):    os.mkdir('./best_model')

    numMSBands = 3 #number of MS bands to extract
    minMSCenters = [400.0, 400.0, 400.0] 
    maxMSCenters = [700.0, 700.0, 700.0]
    minMSFWHMs = [50.0, 50.0, 50.0]
    maxMSFWHMs = [200.0, 200.0, 200.0]
    minMSCenters = None
    maxMSCenters = None
    minMSFWHMs = None
    maxMSFWHMs = None
    batchSize = 128
    valFrequency = 1000
    num_val_batches = 500
    patience = 25

    domain =  [{'name': 'num_layers',          'type': 'discrete',   'domain': (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)},
               {'name': 'num_features',        'type': 'discrete',   'domain': (4,8,16,32,64,128)},
               {'name': 'filter_size',         'type': 'discrete',   'domain': (3,5,7,9,11)},
               {'name': 'learning_rate1',      'type': 'continuous', 'domain': (-5,-1)},
               {'name': 'weight_decay_factor', 'type': 'continuous', 'domain': (-5,-2)},
               {'name': 'learning_rate2',      'type': 'continuous', 'domain': (-5,-1)},
               {'name': 'w1',                  'type': 'continuous', 'domain': (0.,1.)},
               {'name': 'w2',                  'type': 'continuous', 'domain': (0.,1.)}]


    HSBands = np.loadtxt('./dataset/HSBands.csv', delimiter=',').tolist()
    f = objective(HSBands, numMSBands, minMSCenters, maxMSCenters, minMSFWHMs, maxMSFWHMs,\
                 batchSize, valFrequency, num_val_batches, patience)

    myBopt = GPyOpt.methods.BayesianOptimization(f=f,domain=domain,
                                                 model_type='GP_MCMC', acquisition_type='EI_MCMC',
                                                 normalize_Y = True, initial_design_numdata = 5)    
    
    max_iter = 45       ## maximum number of iterations
    max_time = 2.5*24*60*60 #secs  

    myBopt.run_optimization(max_iter,eps=1e-5, max_time=max_time)
    np.savez('./best_model/BO_result_from_optimizer.npz', x=myBopt.X, y=myBopt.Y)


if __name__ == '__main__':
    main()
