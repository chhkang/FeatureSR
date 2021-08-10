import math
import pickle
import torch
import numpy as np
from math import exp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
from torch.utils.tensorboard import SummaryWriter

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def partial_ssim(img1, img2, window, window_size, channel, code, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel) ##E[x]
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel) ##E[y]

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    sigma_12 = torch.sqrt(torch.clamp(sigma1_sq,min=1e-6) * torch.clamp(sigma2_sq,min=1e-6))

    C1 = 0.01**2
    C2 = 0.03**2
    C3 = C2/2
    
    if(code == 'L'):
        ssim_map = (2 * mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
    if(code == 'C'):
        ssim_map = (2 * sigma_12+C2)/(sigma1_sq + sigma2_sq + C2)
    if(code == 'S'):
        ssim_map = (sigma12 + C3)/(sigma_12+C3)
    if(code == 'LC'):
        ssim_map = (2 * mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)*(2*sigma_12+C2)/(sigma1_sq + sigma2_sq + C2)
    if(code == 'CS'):
        ssim_map = (2 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    if(code == 'LS'):
        ssim_map = (2 * mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)*(sigma12 + C3)/(sigma_12+C3)

    return ssim_map.mean()

class Partial_SSIM(nn.Module):
    def __init__(self, code, window_size = 11, size_average = True):
        super(Partial_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel) 
        self.code = code

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return partial_ssim(img1, img2, window, self.window_size, channel, self.code, self.size_average)


class SSIM(nn.Module):
    def __init__(self, window_size = 21, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

writer = SummaryWriter()

class KLD(nn.Module):
    def __init__(self,window_size = 11, log_target = True, reduction='batchmean',epoch=0):
        super(KLD,self).__init__()
        self.window_size = window_size
        self.log_target = log_target
        self.reduction = reduction
        self.epoch = epoch
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        self.epoch = self.epoch+1
        return kld(img1, img2, window, self.window_size, channel, self.log_target, self.reduction, self.epoch)

def kld(img1, img2, window, window_size, channel, log_target, reduction, epoch, nbin=1000):
    max1,min1 = torch.max(img1),torch.min(img1)
    max2,min2 = torch.max(img2),torch.min(img2)
    min = min1 if min1<min2 else min2
    max = max1 if max1>max2 else max2
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel) ##torch.Size([256, 1, 192, 256])

    mu1_ = torch.reshape(img1,(256,192*256))
    mu1_ = [torch.histc(mu1_[x,:],bins=nbin,min=0,max=0) for x in range(256)]
    mu1_ = torch.stack(mu1_)
    mu1_sm = F.softmax(mu1_,dim=1)

    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel) ##E[y]

    mu2_ = torch.reshape(img2,(256,192*256))
    mu2_ = [torch.histc(mu2_[x,:],bins=nbin,min=0,max=0) for x in range(256)]
    mu2_ = torch.stack(mu2_).contiguous()
    mu2_sm = F.softmax(mu2_,dim=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    for i in range(10):
        writer.add_histogram('target{}'.format(epoch),mu2_sm[i,:],i)
        writer.add_histogram('source{}'.format(epoch),mu1_sm[i,:],i)
        V
    # ssim_map = ((2 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)).mean()
    ssim_map = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))).mean()
    # kld = (mu2_sm*(mu2_sm.log()-mu1_sm.log())).mean().cuda()
    kld = nn.KLDivLoss(size_average=True,reduction=reduction,log_target=log_target)(mu1_sm,mu2_sm)

    if(ssim_map>0.75):
        return kld + 1 - ssim_map
    else:
        return 1 - ssim_map

class JSD(nn.Module):
    def __init__(self,window_size = 11, log_target = True, reduction='batchmean'):
        super(JSD,self).__init__()
        self.window_size = window_size
        self.log_target = log_target
        self.reduction = reduction
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return jsd(img1, img2, window, self.window_size, channel, self.log_target, self.reduction)

def jsd(img1, img2, window, window_size, channel, log_target, reduction):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel) ##torch.Size([256, 1, 192, 256])
    mu1_ = torch.reshape(mu1,(256,192*256))
    mu1_ls = F.softmax(mu1_,dim=1)
    mu1_ls = torch.reshape(mu1_ls,(256,1,192,256))
    
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel) ##E[y]
    mu2_ = torch.reshape(mu2,(256,192*256))
    mu2_ls = F.softmax(mu2_,dim=1)
    mu2_ls = torch.reshape(mu2_ls,(256,1,192,256))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))).mean()

    m = 0.5*(mu1_ls+mu2_ls)
    jsd = 0.0
    jsd += (m*(m.log()-mu1_ls.log())).sum()
    jsd += (m*(m.log()-mu2_ls.log())).sum()
    jsd = (jsd * 0.5).cuda()
    # print(torch.mean(mu1_ls),torch.mean(mu2_ls),jsd)

    if(ssim_map>0.75):
        return jsd + 1 - ssim_map
    else:
        return 1 - ssim_map

class WSD(nn.Module):
    def __init__(self,window_size = 11):
        super(WSD,self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return wsd(img1, img2, window, self.window_size, channel)

def wsd(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel) ##torch.Size([256, 1, 192, 256])
    mu1_ = torch.reshape(img1,(256,192*256))
    mu1_ls = F.softmax(mu1_,dim=1)
    mu1_ls = torch.reshape(mu1_ls,(256,1,192,256))
    
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel) ##E[y]
    mu2_ = torch.reshape(img2,(256,192*256))
    mu2_ls = F.softmax(mu2_,dim=1)
    mu2_ls = torch.reshape(mu2_ls,(256,1,192,256))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    # ssim_map = ((2 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)).mean()
    ssim_map = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))).mean()
    wsd = (mu2_ls*(mu2_ls.log()-mu1_ls.log())).sum().cuda()
    print(torch.mean(mu1_ls),torch.mean(mu2_ls),wsd)
    if(ssim_map>0.75):
        return wsd + 1 - ssim_map
    else:
        return 1 - ssim_map


class HD(nn.Module):
    def __init__(self,window_size = 11):
        super(HD,self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return hd(img1, img2, window, self.window_size, channel)

def hd(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel) ##torch.Size([256, 1, 192, 256])
    mu1_ = torch.reshape(img1,(256,192*256))
    mu1_ls = F.softmax(mu1_,dim=1)
    mu1_ls = torch.reshape(mu1_ls,(256,1,192,256))
    
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel) ##E[y]
    mu2_ = torch.reshape(img2,(256,192*256))
    mu2_ls = F.softmax(mu2_,dim=1)
    mu2_ls = torch.reshape(mu2_ls,(256,1,192,256))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    # ssim_map = ((2 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)).mean()
    ssim_map = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))).mean()
    
    hd = torch.cdist(torch.sqrt(mu1_ls),torch.sqrt(mu2_ls)).sum().cuda()
    print(torch.mean(mu1_ls),torch.mean(mu2_ls),hd)
    if(ssim_map>0.75):
        return hd + 1 - ssim_map
    else:
        return 1 - ssim_map

def print_args(args, args_file=None):
    if args_file is None:
        for k, v in sorted(vars(args).items()):
            print('{} {}'.format(k, v))
    else:
        with open(args_file, 'w') as f:
            for k, v in sorted(vars(args).items()):
                f.write('{} {}\n'.format(k, v))

def print_network(model, name, out_file=None):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    if out_file is None:
        print(name)
        print(model)
        print('The number of parameters: {}'.format(num_params))
    else:
        with open(out_file, 'w') as f:
            f.write('{}\n'.format(name))
            f.write('{}\n'.format(model))
            f.write('The number of parameters: {}\n'.format(num_params))


def save_params(params, param_file):
    with open(param_file, 'wb') as f:
        pickle.dump(params, f)


def load_params(param_file):
    with open(param_file, 'rb') as f:
        return pickle.load(f)


class InfDataLoader():
    def __init__(self, dataset, **kwargs):
        self.dataloader = torch.utils.data.DataLoader(dataset, **kwargs)

        def inf_dataloader():
            while True:
                for data in self.dataloader:
                    image, label = data
                    yield image, label

        self.inf_dataloader = inf_dataloader()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.inf_dataloader)

    def __del__(self):
        del self.dataloader

def psnr(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

class LPIPSLoss(nn.Module):
    def __init__(self):
        super(LPIPSLoss,self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        L_ADV = 1e-3        # Scaling params for the Adv loss
        L_FM = 1            # Scaling params for the feature matching loss
        L_LPIPS = 1e-3      # Scaling params for the LPIPS loss        
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)

        # Image Loss
        image_loss = 1.-self.ssim_loss(out_images, target_images)
        # LPIPS Loss
        import LPIPS.models.dist_model as dm
        model_LPIPS = dm.DistModel()
        model_LPIPS.initialize(model='net-lin',net='alex',use_gpu=True)
        LPIPS_loss, _ = model_LPIPS.forward_pair(out_images.repeat(1,3,1,1),target_images.repeat(1,3,1,1))
        LPIPS_loss = torch.mean(LPIPS_loss)

        return L_FM * image_loss + L_ADV * adversarial_loss + L_LPIPS * LPIPS_loss

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images.repeat(1,3,1,1)), self.loss_network(target_images.repeat(1,3,1,1)))
        # Image Loss
        image_loss = 1.-self.ssim_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

##### Changed ####
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x, y):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        x_tvd = self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

        batch_size = y.size()[0]
        h_x = y.size()[2]
        w_x = y.size()[3]
        count_h = self.tensor_size(y[:, :, 1:, :])
        count_w = self.tensor_size(y[:, :, :, 1:])
        h_tv = torch.pow((y[:, :, 1:, :] - y[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((y[:, :, :, 1:] - y[:, :, :, :w_x - 1]), 2).sum()
        y_tvd = self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

        return torch.abs(x_tvd-y_tvd)

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class TVD(nn.Module):
    def __init__(self,window_size = 11):
        super(TVD,self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return tvd(img1, img2, window, self.window_size, channel)

def tvd(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel) ##torch.Size([256, 1, 192, 256])
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel) ##E[y]
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    tvl = TVLoss().cuda()(mu1,mu2)
    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    print(tvl)

    # ssim_map = ((2 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)).mean()
    ssim_map = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))).mean()
    if(ssim_map>0.75):
        return tvl + 1 - ssim_map
    else:
        return 1 - ssim_map