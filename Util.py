import cv2
import os
import torch
import shutil
import PIL.Image
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage.color import gray2rgb
cv2.setNumThreads(0)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def to_np(x):
    x = x.cpu().numpy()
    if len(x.shape)>3:
        return x[:,0:3,:,:]
    else:
        return x

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 50 epochs"""
    lr = args.lr * (0.9 ** ( (epoch+1) // 50))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def measure_hard_mse(output,target,ths):
    _ = torch.abs(output - target)
    _ = (_ < ths) * 1
    items = _.shape[0] * _.shape[1]
    
    return float(_.sum() / items)
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def img_stack_horizontally(images, sep=5):
    w, h = 0, 0
    for img in images:
        h = max(img.size[1], h)
        w += img.size[0] + sep
    fin = PIL.Image.new('RGBA', (w, h))
    cw = 0
    for img in images:
        fin.paste(img, (cw, 0))
        cw += img.size[0] + sep
    return fin

def img_stack_vertically(images, sep=5):
    w, h = 0, 0
    for img in images:
        w = max(img.size[0], w)
        h += img.size[1] + sep
    fin = PIL.Image.new('RGBA', (w, h))
    ch = 0
    for img in images:
        fin.paste(img, (0, ch))
        ch += img.size[1] + sep
    return fin