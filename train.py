#============ Custom tensorboard logging ============#
from TbLogger import Logger

#============ Basic imports ============#
import pickle
import gc
# import cv2
import copy
import os
import time
import tqdm
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
from PIL import Image
# from skimage.io import imsave,imread

# set no multi-processing for cv2 to avoid collisions with data loader
# cv2.setNumThreads(0)

#============ PyTorch imports ============#
import torch
import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
# from torch.nn import Sigmoid

#============ Custom classes ============#
from VAE import VAE, VAEBaseline, VAESimplifiedFC, VAE1N, VAEBaselineConv
from VAELoss import VAELossView as VAELoss
from VAELoss import loss_function
from pytorch_ssim import SSIM
from FMNISTDataset import FMNISTDataset
from Util import str2bool,restricted_float,to_np,save_checkpoint,AverageMeter,img_stack_horizontally,img_stack_vertically

parser = argparse.ArgumentParser(description='VAE FMNIST example')

# ============ basic params ============#
parser.add_argument('--epochs',              default=30,            type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch',         default=0,             type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size',          default=64,            type=int, help='mini-batch size (default: 64)')
parser.add_argument('--seed',                default=42,            type=int, help='random seed (default: 42)')

# ============ data loader and model params ============#
parser.add_argument('--model_multiplier',    default=2.0,           type=float, help='model size multiplier')
parser.add_argument('--do_augs',             default=False,         type=str2bool, help='Whether to use augs')
parser.add_argument('--model_type',          default='fcn',         type=str, help='fcn or fc or fcs')
parser.add_argument('--dataset_type',        default='fmnist',      type=str, help='mnist of fmnist')
parser.add_argument('--latent_space_size',   default=10,            type=int, help='the size of the latent space')

# ============ optimization params ============#
parser.add_argument('--lr',                  default=1e-3,          type=float, help='initial learning rate')
parser.add_argument('--m1',                  default=5,             type=int, help='lr decay milestone 1')
parser.add_argument('--m2',                  default=20,            type=int, help='lr decay milestone 2')

parser.add_argument('--optimizer',           default='adam',        type=str, help='model optimizer')
parser.add_argument('--do_running_mean',     default=False,         type=str2bool, help='Whether to use running mean for loss')
parser.add_argument('--img_loss_weight',     default=1.0,           type=float, help='image reconstruction loss part')
parser.add_argument('--kl_loss_weight',      default=1.0,           type=float, help='kl divergence part')
parser.add_argument('--image_loss_type',     default='bce',         type=str, help='bce, mse or ssim')
parser.add_argument('--ssim_window_size',    default=5,             type=int, help='ssim_window_size')
parser.add_argument('--split_filter',        default=3,             type=int, help='where to split weights for mu and logvar')


# ============ logging params and utilities ============#
parser.add_argument('--print-freq',          default=10,            type=int, help='print frequency (default: 10)')
parser.add_argument('--lognumber',           default='test_model',  type=str, help='text id for saving logs')
parser.add_argument('--tensorboard',         default=False,         type=str2bool, help='Use tensorboard to for loss visualization')
parser.add_argument('--tensorboard_images',  default=False,         type=str2bool, help='Use tensorboard to see images')
parser.add_argument('--resume',              default='',            type=str, help='path to latest checkpoint (default: none)')

# ============ other params ============#
parser.add_argument('--no_cuda',             dest='no_cuda',       action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--predict',             dest='predict',       action='store_true', help='generate prediction masks')
parser.add_argument('--predict_train',       dest='predict_train', action='store_true', help='generate prediction masks')
parser.add_argument('--evaluate',            dest='evaluate',      action='store_true', help='just evaluate')

train_minib_counter = 0
valid_minib_counter = 0
best_loss = 100000000

args = parser.parse_args()
print(args)

# PyTorch 0.4 compatibility
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

# Set the Tensorboard logger
if args.tensorboard or args.tensorboard_images:
    if not (args.predict or args.predict_train):
        logger = Logger('./tb_logs/{}'.format(args.lognumber))
    else:
        logger = Logger('./tb_logs/{}'.format(args.lognumber + '_predictions'))


def main():
    global args, best_loss
    global logger
    global device, kwargs
  
    if args.model_type == 'fcn':
        filter_list = [1,
                       int(args.model_multiplier*4),
                       int(args.model_multiplier*8),
                       int(args.model_multiplier*16),
                       int(args.model_multiplier*32),
                       int(args.model_multiplier*64),
                       10]

        print('Model filter sizes list is {}'.format(filter_list))


        model = VAE(filters          =filter_list,
                    dilations        =[1, 1, 1, 1, 1, 1], 
                    paddings         =[0, 0, 0, 0, 0, 0],
                    strides          =[1, 1, 2, 1, 2, 2],
                    decoder_kernels  =[3, 4, 4, 4, 4, 4],
                    decoder_paddings =[1, 0, 0, 0, 0, 0],
                    decoder_strides  =[1, 1, 1, 2, 2, 1],
                    split_filter     = args.split_filter).to(device)
        
        print(model)
    elif args.model_type == 'fcns_1n':
        filter_list = [1,
                       int(args.model_multiplier*4),
                       int(args.model_multiplier*8),
                       int(args.model_multiplier*16),
                      ]
        
        print('Model filter sizes list is {}'.format(filter_list))
        
        model = VAE1N(filters       =filter_list,
                    dilations         =[1, 1, 1],
                    paddings          =[1, 1, 1],
                    strides           =[2, 2, 2],
                    decoder_kernels   =[4, 4, 3],
                    decoder_paddings  =[1, 1, 1], 
                    decoder_strides   =[2, 2, 2],
                    latent_space_size = 10).to(device)
    elif args.model_type == 'fcns':
        model = VAESimplifiedFC().to(device)        
    elif args.model_type == 'fc':
        model = VAEBaseline(latent_space_size=args.latent_space_size).to(device)
    elif args.model_type == 'fc_conv':
        model = VAEBaselineConv(latent_space_size=args.latent_space_size).to(device)
        
    if args.optimizer.startswith('adam'):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     # Only finetunable params
                                     lr=args.lr)
    elif args.optimizer.startswith('rmsprop'):
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                        # Only finetunable params
                                        lr=args.lr)
    elif args.optimizer.startswith('sgd'):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    # Only finetunable params
                                    lr=args.lr)
    else:
        raise ValueError('Optimizer not supported')        
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])            
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))      
    
    if args.predict:
        pass
    elif args.evaluate:
        pass
    else:
        if args.dataset_type=='fmnist':
            train_dataset = FMNISTDataset(mode = 'train',
                                          random_state = args.seed,
                                          use_augs = args.do_augs)

            val_dataset = FMNISTDataset(mode = 'val',
                                        random_state = args.seed,
                                        use_augs = False)    

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,        
                shuffle=True,
                drop_last=False,
                **kwargs)

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,        
                shuffle=True,            
                drop_last=False,
                **kwargs)
        elif args.dataset_type=='mnist':        
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.ToTensor()),
                batch_size=args.batch_size, shuffle=True, **kwargs)

            val_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
                batch_size=args.batch_size, shuffle=True, **kwargs)
        
        criterion = VAELoss(use_running_mean=args.do_running_mean,
                    image_loss_type=args.image_loss_type,
                    image_loss_weight=args.img_loss_weight,
                    kl_loss_weight=args.kl_loss_weight,
                    ssim_window_size=args.ssim_window_size,
                    latent_space_size=args.latent_space_size).to(device)
        
        # criterion = loss_function

        ssim = SSIM(window_size = args.ssim_window_size,
                    size_average = True).to(device)

        scheduler = MultiStepLR(optimizer, milestones=[args.m1,args.m2], gamma=0.1)  

        for epoch in range(args.start_epoch, args.epochs):

            # train for one epoch
            train_loss,train_img_loss,train_kl_loss,train_ssim = train(train_loader,
                                                            model,
                                                            criterion,
                                                            ssim,
                                                            optimizer,
                                                            epoch)
            
            # evaluate on validation set
            val_loss,val_img_loss,val_kl_loss,val_ssim = validate(val_loader,
                                                         model,
                                                         criterion,
                                                         ssim)
            
            scheduler.step()

            #============ TensorBoard logging ============#
            # Log the scalar values        
            if args.tensorboard:
                info = {
                    'eph_tr_loss': train_loss,
                    'eph_tr_ssim': train_ssim,
                    
                    'eph_val_loss': val_loss,
                    'eph_val_ssim': val_ssim,                    
                    
                    'eph_tr_img_loss': train_img_loss,
                    'eph_tr_kl_loss': train_kl_loss,

                    'eph_val_img_loss': val_img_loss,
                    'eph_val_kl_loss': val_kl_loss,                     
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)                     

            # remember best prec@1 and save checkpoint
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                },
                is_best,
                'weights/{}_checkpoint.pth.tar'.format(str(args.lognumber)),
                'weights/{}_best.pth.tar'.format(str(args.lognumber))
            )
   
def train(train_loader,
          model,
          criterion,
          ssim,
          optimizer,
          epoch):
                                            
    global train_minib_counter
    global logger
        
    # scheduler.batch_step()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    img_losses = AverageMeter()
    kl_losses = AverageMeter()
    ssims = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (input, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().to(device)

        out, mu, logvar = model(input)
        
        loss,image_loss,kl_loss = criterion(out,
                                            input,
                                            mu,
                                            logvar)
        
        ssim_ = ssim(out.view(-1,1,28,28), input.view(-1,1,28,28))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()        

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        img_losses.update(image_loss.item(), input.size(0))
        kl_losses.update(kl_loss.item(), input.size(0))
        ssims.update(ssim_.item(), input.size(0))        
        
        # log the current lr
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                                            
        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'train_loss': losses.val,
                'train_img_loss': img_losses.val,
                'train_kl_loss': kl_losses.val,
                'train_ssim': ssims.val,
                'train_lr': current_lr,      
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, train_minib_counter)                

        train_minib_counter += 1

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'REC  {img_losses.val:.4f} ({img_losses.avg:.4f})\t'
                  'KL   {kl_losses.val:.4f} ({kl_losses.avg:.4f})\t'
                  'SSIM {ssims.val:.4f} ({ssims.avg:.4f})\t'.format(
                   epoch,i, len(train_loader),
                   batch_time=batch_time,data_time=data_time,
                   loss=losses,img_losses=img_losses,kl_losses=kl_losses,
                   ssims=ssims))

    print(' * Avg Train Loss  {loss.avg:.4f}'.format(loss=losses))
    print(' * Avg Train SSIM  {ssims.avg:.4f}'.format(ssims=ssims))
            
    return losses.avg,img_losses.avg,kl_losses.avg,ssims.avg

def validate(val_loader,
             model,
             criterion,
             ssim,
             ):
                                
    global valid_minib_counter
    global logger
    
    # scheduler.batch_step()    
    batch_time = AverageMeter()

    losses = AverageMeter()
    img_losses = AverageMeter()
    kl_losses = AverageMeter()
    
    ssims = AverageMeter()    
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    
    with torch.no_grad():
        for i, (input, _) in enumerate(val_loader):
            input = input.float().to(device)
            # compute output
            out, mu, logvar = model(input)

            loss,image_loss,kl_loss = criterion(out,
                                                input,
                                                mu,
                                                logvar)

            ssim_ = ssim(out.view(-1,1,28,28), input.view(-1,1,28,28))

            #============ TensorBoard logging ============#                                            
            if args.tensorboard_images:
                if i % (args.print_freq*10) == 0:
                    
                    n = min(input.size(0), 40)
                    
                    row1 = img_stack_horizontally([Image.fromarray(np.uint8(_*255)) for _ in input[:n,0].cpu().numpy()])
                    row2 = img_stack_horizontally([Image.fromarray(np.uint8(_*255)) for _ in out.view(args.batch_size, 1, 28, 28)[:n,0].cpu().numpy()])
                    panno = img_stack_vertically([row1, row2])

                    # save_image(comparison.cpu(),
                    #         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

                    # (66, 1320, 4)
                    
                    info = {
                        'panno': np.array(panno)[np.newaxis,:,:,0:3]
                    }
                    for tag, images in info.items():
                        logger.image_summary(tag, images, valid_minib_counter)


            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            img_losses.update(image_loss.item(), input.size(0))
            kl_losses.update(kl_loss.item(), input.size(0))
            ssims.update(ssim_.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #============ TensorBoard logging ============#
            # Log the scalar values        
            if args.tensorboard:
                info = {
                    'val_loss': losses.val,
                    'val_img_loss': img_losses.val,
                    'val_kl_loss': kl_losses.val,
                    'val_ssim': ssims.val,
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, valid_minib_counter)            

            valid_minib_counter += 1

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time  {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss  {loss.val:.4f} ({loss.avg:.4f})\t'
                      'SSIM  {ssims.val:.4f} ({ssims.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time,
                          loss=losses,ssims=ssims))
                
    print(' * Avg Val  Loss  {loss.avg:.4f}'.format(loss=losses))
    print(' * Avg Val  SSIM  {ssims.avg:.4f}'.format(ssims=ssims))

    return losses.avg, img_losses.avg, kl_losses.avg,ssims.avg

if __name__ == '__main__':
    main()