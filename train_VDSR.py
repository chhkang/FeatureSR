import argparse, os
import cv2
import torch
import numpy as np
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.carn import CARN
from models.vdsr import vdsr
from dataset import get_data_loader

# Training settings
parser = argparse.ArgumentParser(description="PyTorch FeatureSR")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=20, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=10, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--data_path", type=str, default = "./dataset")
parser.add_argument("--rescale_factor", type=int, default=8, help="rescale factor for using in training")
parser.add_argument("--feature_type", type=str, default='p2', help="Feature type for usingin training")
parser.add_argument("--clip", type=float, default=10.0, help="Clipping Gradients. Default=10.0")
parser.add_argument("--group", type=int, default=1)
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--loss_type", type=str, choices= ["MSE", "L1", "SmoothL1","vgg_loss","ssim_loss","adv_loss","lpips"], default='ssim_loss', help="loss type in training")

def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)
    if(opt.feature_type == 'p2'):
        shape = (-1,192,256,1)
    elif(opt.feature_type == 'p3'):
        shape = (-1,96,128,1)
    elif(opt.feature_type == 'p4'):
        shape = (-1,48,64,1)
    elif(opt.feature_type == 'p5'):
        shape = (-1,24,32,1)
    elif(opt.feature_type == 'p6'):
        shape = (-1,12,16,1)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
    print("===> Loading datasets")
    assert opt.rescale_factor%2==0, 'only for 2,4,8,16,32'
    train_loader,test_loader = get_data_loader(opt.data_path,opt.feature_type,opt.rescale_factor,opt.batchSize,opt.threads)

    print("===> Building model")
    netG = vdsr(scale=opt.rescale_factor)
    # netG.apply(models.ops.weights_init)
    netG.train()
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0,0.99))    
    criterion = utils.SSIM()

    if opt.loss_type == 'GAN':
        netD = Discriminator(opt.feature_type).cuda()
        # netD.apply(models.weights_init)
        netD.train()
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0,0.99))

    print("===> Setting GPU")
    if cuda:
        netG = netG.cuda()
        criterion = criterion.cuda()
        if opt.loss_type =='GAN':
            netD = netD.cuda()
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict(),strict = False)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    print("===> Training")
    best_acc = 0
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print("Epoch={}, lr={}".format(epoch, optimizerG.param_groups[0]["lr"]))
        netG.train()
        n = random.randint(0,opt.n_critic - 1)
        for batch_idx, batch in enumerate(train_loader, 1):
            input, target = Variable(batch[0]), Variable(batch[1], requires_grad=True)
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            
            output = netG(input) ##output = SR
            if opt.loss_type=='GAN':
                loss_D = -torch.mean(netD(target)+torch.mean(netD(output)))
                loss_D.backward()
                nn.utils.clip_grad_norm_(netD.parameters(),opt.clip) 
                optimizerD.step()
                if(batch_idx % opt.n_critic == n):
                    optimizerG.zero_grad()
                    output = netG(input)
                    loss_G = -torch.mean(netD(output))
                    loss_focal = criterion(output,target)
                    loss_G = loss_G + loss_focal
                    loss_G.backward()
                    nn.utils.clip_grad_norm_(netG.parameters(),opt.clip) 
                    optimizerG.step()
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Focal loss: %f]"% (epoch, opt.nEpochs, batch_idx % len(train_loader), len(train_loader), loss_D.item(), loss_G.item(),loss_focal.item()))
            else:
                loss = 1.- criterion(output,target)
                optimizerG.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(netG.parameters(),opt.clip) 
                optimizerG.step()
                if batch_idx%30 == 0:
                    print("===> Epoch[{}]({}/{}): SSIM Loss: {:.5}".format(epoch, batch_idx, len(train_loader), loss.item()))
        if not os.path.exists('./result/{}/VDSR_{}x{}'.format(opt.loss_type, opt.feature_type,opt.rescale_factor)):
            os.makedirs('./result/{}/VDSR_{}x{}'.format(opt.loss_type, opt.feature_type,opt.rescale_factor))

        ### valid ###
        PSNR_PREDICTED = 0
        psnr_BICUBIC = 0
        for batch_idx, (lr_f, hr_f) in enumerate(test_loader):
            lr_f = lr_f.cuda()
            sr_f = torch.clamp(netG(lr_f),0,1)
            sr_f = torch.reshape(sr_f,shape)*255

            for i in range(0,8):
                low = sr_f[8*i,:,:,:]
                for j in range(1,8):
                    low = torch.cat((low,sr_f[8*i+j,:,:,:]),0)
                if(i == 0):
                    pic = low
                else:
                    pic = torch.cat((pic,low),1)
            sr_f = pic.cpu().detach().numpy() ###256x1x50x68
            sr_f = sr_f.astype(np.uint8)

            lr_f = torch.reshape(lr_f, (-1,int(shape[1]/opt.rescale_factor),int(shape[2]/opt.rescale_factor),1))*255
            for i in range(0,8):
                low = lr_f[8*i,:,:,:]
                for j in range(1,8):
                    low = torch.cat((low,lr_f[8*i+j,:,:,:]),0)
                if(i == 0):
                    pic = low
                else:
                    pic = torch.cat((pic,low),1)
            lr_f = pic.cpu().detach().numpy() ###256x1x50x68
            lr_f = lr_f.astype(np.uint8)

            hr_f = torch.reshape(hr_f,shape)*255
            for i in range(0,8):
                low = hr_f[8*i,:,:,:]
                for j in range(1,8):
                    low = torch.cat((low,hr_f[8*i+j,:,:,:]),0)
                if(i == 0):
                    pic = low
                else:
                    pic = torch.cat((pic,low),1)
            hr_f = pic.cpu().detach().numpy() ###256x1x50x68
            hr_f = hr_f.astype(np.uint8)
            if(batch_idx<10):
                if(epoch == 1):
                    cv2.imwrite("./result/{}/VDSR_{}x{}/LR_{}.png".format(opt.loss_type, opt.feature_type,opt.rescale_factor,batch_idx),lr_f)
                    cv2.imwrite("./result/{}/VDSR_{}x{}/HR_{}.png".format(opt.loss_type, opt.feature_type,opt.rescale_factor,batch_idx),hr_f)
                cv2.imwrite("./result/{}/VDSR_{}x{}/SR_{}.png".format(opt.loss_type, opt.feature_type,opt.rescale_factor,batch_idx),sr_f)
            PSNR_PREDICTED += utils.psnr(hr_f.astype(float),sr_f.astype(float))

        print("Evaluation ===> Epoch[{}] : PSNR_predicted : {:.5f}".format(epoch, PSNR_PREDICTED/len(test_loader)))

        if(PSNR_PREDICTED/len(test_loader)>best_acc):
            best_acc = PSNR_PREDICTED/len(test_loader)
            save_checkpoint(netG, epoch)

def save_checkpoint(model, epoch):
    print('Saving Networks..')
    model_out_path = './result/{}/VDSR_{}x{}/model.pth'.format(opt.loss_type, opt.feature_type,opt.rescale_factor)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
if __name__ == "__main__":
    main()