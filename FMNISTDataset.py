import os
import cv2
import time
import random 
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

cv2.setNumThreads(0)

class FMNISTDataset(data.Dataset):
    def __init__(self,
                 num_folds = 5,
                 fold = 0,
                 mode = 'train',
                 random_state = 42,
                 use_augs = False,
                ):
        
        self.X, self.y, self.X_test, self.y_test = self.get_fashion_mnist()
        self.fold = fold            
        self.num_folds = num_folds
        self.mode = mode
        self.random_state = random_state
        self.mean = self.X.reshape(60000, 28, 28).mean()/255
        self.std = self.X.reshape(60000, 28, 28).std()/255
        self.use_augs = use_augs
        
        skf = StratifiedKFold(n_splits = self.num_folds,
                              shuffle = True,
                              random_state = self.random_state)
        
        f1, f2, f3, f4, f5 = skf.split(self.X,self.y)
        
        self.folds = [f1, f2, f3, f4, f5]
        self.train_idx = self.folds[self.fold][0]
        self.val_idx = self.folds[self.fold][1] 
    def get_fashion_mnist(self):
        if not os.path.isfile('train-images-idx3-ubyte'):
            os.system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz')
            os.system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz')
            os.system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz')
            os.system('wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz')
            os.system('gunzip *.gz')

        with open('train-images-idx3-ubyte', 'rb') as f:
            X = np.frombuffer(f.read(), dtype=np.uint8, offset=16).copy()
            X = X.reshape((60000, 28*28))

        with open('train-labels-idx1-ubyte', 'rb') as f:
            y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

        with open('t10k-images-idx3-ubyte', 'rb') as f:
            X_test = np.frombuffer(f.read(), dtype=np.uint8, offset=16).copy()
            X_test = X_test.reshape((10000, 28*28))

        with open('t10k-labels-idx1-ubyte', 'rb') as f:
            y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

        return X, y, X_test, y_test
    def stratified_sample(self,
                          images_per_class):
        
        all_targets = list(self.y)
        samples = []
        targets = []
        
        for target_value in set(all_targets):
            indices = [i for i, e in enumerate(all_targets) if e == target_value]
            random.shuffle(indices)
            # produce max images_per_class
            indices = indices[:images_per_class]
            samples.append(self.X[np.asarray(indices)])
            targets.append(self.y[np.asarray(indices)])
        
        return np.vstack(samples),np.vstack(targets).reshape(-1)
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)
        elif self.mode == 'eval_train':
            return self.X.shape[0]
        elif self.mode == 'eval_test':
            return self.X_test.shape[0]        
    def __getitem__(self, idx):
        if self.mode == 'train':
            return_tuple = (self.preprocess_img(Image.fromarray(self.X[self.train_idx[idx]].reshape(28, 28))),
                            self.y[self.train_idx[idx]])
        elif self.mode == 'val':
            return_tuple = (self.preprocess_img(Image.fromarray(self.X[self.val_idx[idx]].reshape(28, 28))),
                            self.y[self.val_idx[idx]])
        elif self.mode == 'eval_train':
            return_tuple = (self.preprocess_img(Image.fromarray(self.X[idx].reshape(28, 28))),
                            self.y[idx])
        elif self.mode == 'eval_test':
            return_tuple = (self.preprocess_img(Image.fromarray(self.X_test[idx].reshape(28, 28))),
                            self.y_test[idx])               
        return return_tuple 
    def preprocess_img(self,
                       img):
        if self.use_augs == False:
            preprocessing = transforms.Compose([
                            transforms.CenterCrop(28),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[self.mean],
                            #                     std=[self.std]),
                            ])            
        else:
            # do some naive augs
            preprocessing = transforms.Compose([
                            transforms.CenterCrop(28),
                            transforms.RandomHorizontalFlip(p=0.25),
                            transforms.RandomRotation(degrees=30),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[self.mean],
                                                 std=[self.std]),
                            ])  
        return preprocessing(img).numpy()            
    def reverse_normalize(tensor, mean, std):
        '''reverese normalize to convert tensor -> PIL Image'''
        tensor_copy = tensor.clone()
        for t, m, s in zip(tensor_copy, mean, std):
            t.div_(s).sub_(m)
        return tensor_copy
    def tensor2img(tensor, on_cuda=True):
        tensor = reverse_normalize(tensor, REVERSE_MEAN, REVERSE_STD)
        # clipping
        tensor[tensor > 1] = 1
        tensor[tensor < 0] = 0
        tensor = tensor.squeeze(0)
        if on_cuda:
            tensor = tensor.cpu()
        return transforms.ToPILImage()(tensor)    