import numpy as  np
import scipy as sp
import os
import random
from multiprocessing import Process, Queue
import tqdm

queue_size = 3
num_activeimages = 20

#These need to be changed for other datasts
rootTrainDir = './dataset/train'
rootValidationDir = './dataset/val'
imgExtension = '.npy'

def read_image(fname):
    img = np.load(fname)
    return img.reshape(-1,img.shape[-1])
############################################    
    

class ImageLoader:
    def __init__(self,path, queue):
        self.path = path
        self.queue = queue
        self.ptr = 0
        self.shuffle_idx = np.random.permutation(range(len(self.path)))
  
    def __call__(self):
        while True:
            img = read_image(self.path[self.shuffle_idx[self.ptr]])
            self.ptr = self.ptr + 1 
            if self.ptr >= len(self.path):
                self.ptr = 0
                self.shuffle_idx = np.random.permutation(range(len(self.path)))
            self.queue.put(img)     
     
#############################################

class SpectraLoader:
    def __init__(self, split='train'):

        if split=='train':
            self.chips_path = [os.path.join(rootTrainDir,f) for f in os.listdir(rootTrainDir) if f.endswith(imgExtension)]
        else:
            self.chips_path = [os.path.join(rootValidationDir,f) for f in os.listdir(rootValidationDir) if f.endswith(imgExtension)]
       
        self.image_queue = Queue(queue_size)
        self.loader = ImageLoader(self.chips_path, self.image_queue)
        self.image_process = Process(target=self.loader)
        self.image_process.daemon = True
        self.image_process.start()

        self.activeimages = [self.image_queue.get() for i in range(num_activeimages)]
        self.pixel_shuffle = [ np.random.permutation(self.activeimages[i].shape[0]) for i in range(num_activeimages)] 
        self.pixel_pointer = [ 0 for i in range(num_activeimages)] 

    def __del__(self):
        del self.image_process   

    def get_example(self):
        while True:
            chosen_image = np.random.randint(num_activeimages)
            img = self.activeimages[chosen_image]
  
            r = self.pixel_shuffle[chosen_image][  self.pixel_pointer[chosen_image] ] 
            spectrum = img[r,:]
 
            self.pixel_pointer[chosen_image] += 1
            if self.pixel_pointer[chosen_image] >= img.shape[0]:
                self.activeimages[chosen_image] = self.image_queue.get()
                self.pixel_pointer[chosen_image] = 0
                self.pixel_shuffle[chosen_image] = np.random.permutation(range(img.shape[0]))

            if not np.all(spectrum==0):
                break
        return spectrum

    def get_mixed_example(self):
        mixed_fraction = 0.3
        if np.random.rand() < mixed_fraction:
            num_endmembers = np.random.randint(2,6);
            abundances = np.random.rand(num_endmembers)
            abundances = abundances / abundances.sum()
            spectrum = 0
            for i in range(num_endmembers):
                spectrum = abundances[i] * self.get_example()
            return spectrum 
        else:
            return self.get_example()

    def get_batch(self, batch_size, mixed = False):
        examples = []
        for i in range(batch_size):
            if mixed:
                an_example =  self.get_mixed_example()
            else:
                an_example =  self.get_example()
            examples.append(an_example)
        return np.array(examples)
    
