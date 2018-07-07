import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_ssim import SSIM as SSIMLoss

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

class VAELossView(nn.Module):
    def __init__(self,
                 use_running_mean=False,
                 image_loss_type='bce',# 'bce','mse' or ssim
                 image_loss_weight=1.0,
                 kl_loss_weight=1.0,
                 ssim_window_size=5,
                 eps=1e-10,
                 gamma=0.9,
                 latent_space_size=10
                 ):
        super().__init__()

        if image_loss_type=='bce':
            self.image_loss = nn.BCELoss(size_average=False)
        elif image_loss_type=='mse':
            self.image_loss = nn.MSELoss(size_average=False)
        elif image_loss_type=='ssim':
            self.image_loss = SSIMLoss(window_size = ssim_window_size, size_average = False)
            
        self.image_loss_type = image_loss_type
        self.use_running_mean = use_running_mean            
        self.image_loss_weight = image_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.eps = eps
        self.gamma = gamma 
        self.latent_space_size = latent_space_size
        
        if self.use_running_mean == True:
            self.register_buffer('running_image_loss', torch.zeros(1))
            self.register_buffer('running_kl_loss', torch.zeros(1))
            self.reset_parameters()

    def reset_parameters(self):
        self.running_image_loss.zero_()        
        self.running_kl_loss.zero_()            

    def forward(self, 
                outputs,
                targets,
                mu,
                logvar):
        
        if self.image_loss_type=='ssim':
            image_loss = 1-self.image_loss(outputs, targets)
            outputs = outputs.view(-1, 784)
            targets = targets.view(-1, 784)
            mu = mu.view(-1, self.latent_space_size)
            logvar = logvar.view(-1, self.latent_space_size)            
        else:
            outputs = outputs.view(-1, 784)
            targets = targets.view(-1, 784) 
            mu = mu.view(-1, self.latent_space_size)
            logvar = logvar.view(-1, self.latent_space_size)
            
            image_loss = self.image_loss(outputs, targets)
            
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.use_running_mean == False:
            imw = self.image_loss_weight
            kmw = self.kl_loss_weight
        else:
            self.running_image_loss = self.running_image_loss * self.gamma + image_loss.data * (1 - self.gamma)        
            self.running_kl_loss = self.running_kl_loss * self.gamma + kl_loss.data * (1 - self.gamma)

            im = float(self.running_image_loss)
            km = float(self.running_kl_loss)

            imw = 1 - im / (im + km)
            kmw = 1 - km / (im + km)
                
        loss = image_loss * imw + kl_loss * kmw
        
        return loss,image_loss,kl_loss

class VAELoss(nn.Module):
    def __init__(self,
                 use_running_mean=False,
                 image_loss_type='bce',# 'bce','mse' or ssim
                 image_loss_weight=1.0,
                 kl_loss_weight=1.0,
                 ssim_window_size=5,
                 eps=1e-10,
                 gamma=0.9
                 ):
        super().__init__()

        if image_loss_type=='bce':
            self.image_loss = nn.BCELoss(size_average=True)
        elif image_loss_type=='mse':
            self.image_loss = nn.MSELoss(size_average=True)
        elif image_loss_type=='ssim':
            self.image_loss = SSIMLoss(window_size = ssim_window_size, size_average = True)
            
        self.image_loss_type = image_loss_type
        self.use_running_mean = use_running_mean            
        self.image_loss_weight = image_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.eps = eps
        self.gamma = gamma 
        
        if self.use_running_mean == True:
            self.register_buffer('running_image_loss', torch.zeros(1))
            self.register_buffer('running_kl_loss', torch.zeros(1))
            self.reset_parameters()

    def reset_parameters(self):
        self.running_image_loss.zero_()        
        self.running_kl_loss.zero_()            

    def forward(self, 
                outputs,
                targets,
                mu,
                logvar):
        
        # inputs and targets are assumed to be BxCxWxH
        assert len(outputs.shape) == len(targets.shape)
        # assert that B, W and H are the same
        assert outputs.size(0) == targets.size(0)
        assert outputs.size(2) == targets.size(2)
        assert outputs.size(3) == targets.size(3)
        
        assert mu.size(1) == 10
        assert logvar.size(1) == 10
        assert mu.size(2) == 1
        assert logvar.size(2) == 1
        assert mu.size(3) == 1
        assert logvar.size(3) == 1    
        
        if self.image_loss_type=='ssim':
            image_loss = 1-self.image_loss(outputs, targets)
        else:
            image_loss = self.image_loss(outputs, targets)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.use_running_mean == False:
            imw = self.image_loss_weight
            kmw = self.kl_loss_weight
        else:
            self.running_image_loss = self.running_image_loss * self.gamma + image_loss.data * (1 - self.gamma)        
            self.running_kl_loss = self.running_kl_loss * self.gamma + kl_loss.data * (1 - self.gamma)

            im = float(self.running_image_loss)
            km = float(self.running_kl_loss)

            imw = 1 - im / (im + km)
            kmw = 1 - km / (im + km)
                
        loss = image_loss * imw + kl_loss * kmw
        
        return loss,image_loss,kl_loss