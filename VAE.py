import torch
from torch import nn
from torch.nn import functional as F

class EncoderBlock(nn.Module):
    def __init__(self,
                 in_,
                 out,
                 padding=1,
                 dilation=1,
                 stride=1,
                 activation=nn.ReLU(inplace=True),
                 kernel_size=3):
        super().__init__()
        
        self.conv = nn.Conv2d(in_,
                     out,
                     kernel_size,
                     padding=padding,
                     dilation=dilation,
                     stride=stride)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        # print(x.shape)
        return x
     
class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 padding=1,
                 dilation=0,
                 stride=2,
                 activation=nn.ReLU(inplace=True)
                ):

        super(DecoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation),
            activation
        )
    def forward(self, x):
        x = self.block(x)
        # print(x.shape)
        return x 
    
class VAE(nn.Module):
    def __init__(self,
                 filters          =[1, 16, 32, 64, 32, 16, 10],
                 dilations        =[1, 1, 1, 1, 1, 1],
                 paddings         =[0, 0, 0, 0, 0, 0],
                 strides          =[1, 1, 2, 1, 2, 2],
                 decoder_kernels  =[3, 4, 4, 4, 4, 4],
                 decoder_paddings =[1, 0, 0, 0, 0, 0],
                 decoder_strides  =[1, 1, 1, 2, 2, 1],
                 split_filter = 3):
        super().__init__()
        
        modules = []
        modules_mu_encoder = []
        modules_logvar_encoder = []
        decoder_modules = []
        
        for i in range(0,len(filters)-1):
            modules_encoder = self.add_conv_block(modules,
                                                  EncoderBlock,
                                                  filters[i],
                                                  filters[i+1],
                                                  dilations[i],
                                                  paddings[i],
                                                  strides[i])
            
            modules_mu_encoder = self.add_conv_block(modules_mu_encoder,
                                                  EncoderBlock,
                                                  filters[i],
                                                  filters[i+1],
                                                  dilations[i],
                                                  paddings[i],
                                                  strides[i])
            
            modules_logvar_encoder = self.add_conv_block(modules_logvar_encoder,
                                                  EncoderBlock,
                                                  filters[i],
                                                  filters[i+1],
                                                  dilations[i],
                                                  paddings[i],
                                                  strides[i])             
            
            decoder_modules = self.add_conv_block(decoder_modules,
                                          DecoderBlock,
                                          filters[i+1],
                                          filters[i],
                                          dilations[i],
                                          decoder_paddings[i],
                                          decoder_strides[i],
                                          decoder_kernels[i])
        
        # make sure the weights are shared
        self.encoder = nn.Sequential(*modules[:-split_filter]) 
        self.mu_encoder = nn.Sequential(*modules_mu_encoder[-split_filter:])
        self.logvar_encoder = nn.Sequential(*modules_logvar_encoder[-split_filter:])        

        self.decoder = nn.Sequential(*reversed(decoder_modules))
        
        print('Total model parameters {}'.format(self.count_parameters()))
        assert self.count_parameters() < 120000        
            
    def encode(self,x):
        x = self.encoder(x)
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        return mu,logvar

    def decode(self, z):
        z = self.decoder(z)
        return F.sigmoid(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        # print(mu.shape,logvar.shape)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def add_conv_block(self,
                       modules,
                       block,
                       filters_in,
                       filters_out,
                       dilation,
                       padding,
                       stride,
                       kernel_size=3
                       ):
       
        modules.append(block(filters_in,
                             filters_out,
                             padding=padding,
                             dilation=dilation,
                             stride=stride,
                             kernel_size=kernel_size))
        
        return modules
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
class VAEBaselineView(nn.Module):
    def __init__(self):
        super(VAEBaseline, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        
        print('Total model parameters {}'.format(self.count_parameters()))

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z).view(-1, 1, 28, 28), mu.view(-1, 10, 1, 1), logvar.view(-1, 10, 1, 1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class VAEBaseline(nn.Module):
    def __init__(self,
                 latent_space_size=10):
        super(VAEBaseline, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_space_size)
        self.fc22 = nn.Linear(400, latent_space_size)
        self.fc3 = nn.Linear(latent_space_size, 400)
        self.fc4 = nn.Linear(400, 784)
        
        print('Total model parameters {}'.format(self.count_parameters()))
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)    
    
class VAESimplified(nn.Module):
    def __init__(self):
        super().__init__()
        
        # make sure the weights are shared
        self.encoder = nn.Sequential(
            nn.Conv2d(1,4,3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4,8,3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8,16,3,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,stride=2),
            nn.ReLU(inplace=True),            
        ) 
        self.mu_encoder = nn.Sequential(
            nn.Conv2d(64,10,3,stride=2),
            nn.ReLU(inplace=True),            
        ) 
        self.logvar_encoder = nn.Sequential(
            nn.Conv2d(64,10,3,stride=2),
            nn.ReLU(inplace=True),          
        )       

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10,64,4,stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,32,4,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,16,4,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,8,4,stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8,4,4,stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4,1,3,stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),              
        ) 
        
        print('Total model parameters {}'.format(self.count_parameters()))
        assert self.count_parameters() < 120000        
            
    def encode(self,x):
        x = self.encoder(x)
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        return mu,logvar

    def decode(self, z):
        z = self.decoder(z)
        return F.sigmoid(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        # print(mu.shape,logvar.shape)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
class VAEView(nn.Module):
    def __init__(self,
                 filters           =[1, 6, 12, 24],
                 dilations         =[1, 1, 1],
                 paddings          =[1, 1, 1],
                 strides           =[2, 2, 2],
                 decoder_kernels   =[4, 4, 3],
                 decoder_paddings  =[1, 1, 1], 
                 decoder_strides   =[2, 2, 2],
                 latent_space_size = 10): 
        super().__init__()

        self.filters = filters
        modules = []
        decoder_modules = []
        
        for i in range(0,len(filters)-1):
            modules = self.add_conv_block(modules,
                                          EncoderBlock,
                                          filters[i],
                                          filters[i+1],
                                          dilations[i],
                                          paddings[i],
                                          strides[i])
            
            decoder_modules = self.add_conv_block(decoder_modules,
                                          DecoderBlock,
                                          filters[i+1],
                                          filters[i],
                                          dilations[i],
                                          decoder_paddings[i],
                                          decoder_strides[i],
                                          decoder_kernels[i])
            
        # make sure the weights are shared
        self.encoder = nn.Sequential(*modules) 
        
        self.mu_encoder = nn.Conv2d(4*4*filters[-1],latent_space_size, kernel_size=3, stride=1, padding=1)
        self.logvar_encoder = nn.Conv2d(4*4*filters[-1],latent_space_size, kernel_size=3,  stride=1, padding=1)        

        self.decoder_transition = EncoderBlock(latent_space_size,4*4*filters[-1])
        self.decoder = nn.Sequential(*reversed(decoder_modules))
        
        print('Total model parameters {}'.format(self.count_parameters()))
        assert self.count_parameters() < 120000        
            
    def encode(self,x):
        x = self.encoder(x)
        x = x.view(x.shape[0],-1,1,1)
        
        # print(x.shape)

        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        
        return mu,logvar

    def decode(self, z):
        z = self.decoder_transition(z)
        z = z.view(-1,self.filters[-1],4,4)
        z = self.decoder(z)
        return F.sigmoid(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        # print(mu.shape,logvar.shape)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def add_conv_block(self,
                       modules,
                       block,
                       filters_in,
                       filters_out,
                       dilation,
                       padding,
                       stride,
                       kernel_size=3
                       ):
       
        modules.append(block(filters_in,
                             filters_out,
                             padding=padding,
                             dilation=dilation,
                             stride=stride,
                             kernel_size=kernel_size))
        
        return modules
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
class VAE1N(nn.Module):
    def __init__(self,
                 filters           =[1, 6, 12, 24],
                 dilations         =[1, 1, 1],
                 paddings          =[1, 1, 1],
                 strides           =[2, 2, 2],
                 decoder_kernels   =[4, 4, 3],
                 decoder_paddings  =[1, 1, 1], 
                 decoder_strides   =[2, 2, 2],
                 latent_space_size = 10): 
        super().__init__()

        self.filters = filters
        modules = []
        decoder_modules = []
        
        for i in range(0,len(filters)-1):
            modules = self.add_conv_block(modules,
                                          EncoderBlock,
                                          filters[i],
                                          filters[i+1],
                                          dilations[i],
                                          paddings[i],
                                          strides[i])
            
            decoder_modules = self.add_conv_block(decoder_modules,
                                          DecoderBlock,
                                          filters[i+1],
                                          filters[i],
                                          dilations[i],
                                          decoder_paddings[i],
                                          decoder_strides[i],
                                          decoder_kernels[i])
            
        # make sure the weights are shared
        self.encoder = nn.Sequential(*modules) 
        
        self.mu_encoder = nn.Conv1d(1,latent_space_size, kernel_size=(4*4*filters[-1]), stride=1, padding=0)
        self.logvar_encoder = nn.Conv1d(1,latent_space_size, kernel_size=(4*4*filters[-1]),  stride=1, padding=0)     

        self.decoder_transition = nn.Sequential(
            nn.ConvTranspose1d(latent_space_size,1, kernel_size=(4*4*filters[-1]),  stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(*reversed(decoder_modules))
        
        print('Total model parameters {}'.format(self.count_parameters()))
        assert self.count_parameters() < 120000        
            
    def encode(self,x):
        x = self.encoder(x)
        x = x.view(x.shape[0],1,-1)
        
        # print(x.shape)

        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        
        # print(mu.shape)
        # print(logvar.shape)
        
        return mu,logvar

    def decode(self, z):
        z = self.decoder_transition(z)
        z = z.view(-1,self.filters[-1],4,4)
        z = self.decoder(z)
        return F.sigmoid(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        # print(mu.shape,logvar.shape)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def add_conv_block(self,
                       modules,
                       block,
                       filters_in,
                       filters_out,
                       dilation,
                       padding,
                       stride,
                       kernel_size=3
                       ):
       
        modules.append(block(filters_in,
                             filters_out,
                             padding=padding,
                             dilation=dilation,
                             stride=stride,
                             kernel_size=kernel_size))
        
        return modules
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu                
        
class VAESimplifiedFC(nn.Module):
    def __init__(self):
        super().__init__()
        
        # make sure the weights are shared
        self.encoder = nn.Sequential(
            nn.Conv2d(1,4,3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4,8,3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8,16,3,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,stride=2),
            nn.ReLU(inplace=True),            
        ) 
        self.mu_encoder = nn.Sequential(
            nn.Linear(64*4*4, 10)          
        ) 
        self.logvar_encoder = nn.Sequential(
            nn.Linear(64*4*4, 10)
        )       
        self.linear_decoder = nn.Sequential(
            nn.Linear(10, 64*4*4),
            nn.ReLU(inplace=True),
        )       
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,16,4,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,8,4,stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8,4,4,stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4,1,3,stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),              
        ) 
        
        print('Total model parameters {}'.format(self.count_parameters()))
        assert self.count_parameters() < 120000        
            
    def encode(self,x):
        x = self.encoder(x)
        x = x.view(-1,64*4*4)
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        return mu,logvar

    def decode(self, z):
        z = self.linear_decoder(z)
        z = z.view(-1,64,4,4)
        z = self.decoder(z)
        return F.sigmoid(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        # print(mu.shape,logvar.shape)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu          
        
class VAEBaselineConv(nn.Module):
    def __init__(self,
                 latent_space_size=10):
        super(VAEBaselineConv, self).__init__()
        self.fc1 = nn.Conv2d(1,32, kernel_size=(28,28), stride=1, padding=0)
        self.fc21 = nn.Conv2d(32,latent_space_size, kernel_size=(1,1), stride=1, padding=0)
        self.fc22 = nn.Conv2d(32,latent_space_size, kernel_size=(1,1), stride=1, padding=0)
        
        self.fc3 = nn.ConvTranspose2d(latent_space_size,118, kernel_size=(1,1),  stride=1, padding=0)
        self.fc4 = nn.ConvTranspose2d(118,1, kernel_size=(28,28),  stride=1, padding=0)
        
        print('Total model parameters {}'.format(self.count_parameters()))
        assert self.count_parameters() < 120000    
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)          