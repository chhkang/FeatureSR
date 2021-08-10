import argparse, os, sys
import cv2
import torch
import models
import numpy as np
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.vdsr import VDSR
from models.edsr import EDSR
from models.carn import CARN,Discriminator
from dataset import get_data_loader

# Training settings
parser = argparse.ArgumentParser(description="PyTorch FeatureSR")
parser.add_argument("--batchSize", type=int, default=256, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=1e-4")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=10, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--model", type=str, choices= ["CARN","EDSR","VDSR"], default="CARN")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--data_path", type=str, default = "./dataset")
parser.add_argument("--rescale_factor", type=int, default=4, help="rescale factor for using in training")
parser.add_argument("--feature_type", type=str, default='p2', help="Feature type for usingin training")
parser.add_argument("--clip", type=float, default=10.0, help="Clipping Gradients. Default=10.0")
parser.add_argument("--group", type=int, default=1)
parser.add_argument("--n_critic", type=int, default=3, help="number of training steps for discriminator per iter")
parser.add_argument("--loss_type", type=str, choices= ["MSE", "L1", "SmoothL1","vgg_loss","ssim_loss","adv_loss","lpips"], default='ssim_loss', help="loss type in training")
parser.add_argument("--code", type=str, help="combination of ssim L,C,S")


def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    file_pth = './result/ssim_loss_{}/CARN_p2x{}'.format(opt.code,opt.rescale_factor)
    if not os.path.exists(file_pth):
        os.makedirs(file_pth)
    sys.stdout = open(os.path.join(file_pth,"log.txt"),'w')

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

    print("===> Building model: {}".format(opt.model))
    if opt.model == 'CARN':
        netG = CARN(scale=opt.rescale_factor)
        # model_path = './result/ssim_loss/CARN_p2x{}/model.pth'.format(opt.rescale_factor)
        # checkpoint = torch.load(model_path)
        # netG.load_state_dict(checkpoint,strict = True)

    elif opt.model == 'EDSR':
        netG = EDSR(scale=opt.rescale_factor)
    elif opt.model == 'VDSR':
        netG = VDSR(scale=opt.rescale_factor)
    netG.apply(models.init_weights)
    netG.train()
    # optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0,0.99))    
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9, 0.999), eps= 1e-8) ##EDSR Exp default setting
    # optimizerG = optim.RMSprop(netG.parameters(),lr=opt.lr)
    pixel_criterion = utils.Partial_SSIM(code=opt.code)

    if opt.loss_type == 'adv_loss':
        add_criterion = utils.GeneratorLoss()
    elif opt.loss_type == 'lpips':
        add_criterion = utils.LPIPSLoss()

    if opt.loss_type == 'adv_loss' or opt.loss_type == 'lpips':
        netD = Discriminator(shape)
        netD.apply(models.init_weights)
        netD.train()
        optimizerD = optim.RMSprop(netD.parameters(),lr=opt.lr)

    print("===> Setting GPU")
    if cuda:
        netG = netG.cuda()
        pixel_criterion = pixel_criterion.cuda()
        if opt.loss_type =='adv_loss' or opt.loss_type == 'lpips':
            netD = netD.cuda()
            add_criterion = add_criterion.cuda()
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
            if (opt.loss_type =='adv_loss' or opt.loss_type == 'lpips'):# and epoch> int(opt.nEpochs/2):
                # (1) Update D network: maximize D(x)-1-D(G(z))
                optimizerD.zero_grad()
                loss_D = 1 - torch.mean(netD(target) + torch.mean(netD(output)))
                loss_D.backward(retain_graph=True)
                optimizerD.step()
                # for p in netD.parameters():
                #     p.data.clamp_(-opt.clip, opt.clip)

                # (2) Update G network: minimize 1-D(G(z)) + Image Loss 
                # if(batch_idx % opt.n_critic == 0):
                optimizerG.zero_grad()
                fake_img = netG(input)
                fake_out = torch.mean(netD(fake_img))
                loss_G = add_criterion(fake_out,fake_img,target)
                loss_G.backward()
                optimizerG.step()
                for p in netG.parameters():
                    p.data.clamp_(-opt.clip, opt.clip)

                nn.utils.clip_grad_norm_(netG.parameters(),opt.clip) 
                if batch_idx%30 == 0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"% (epoch, opt.nEpochs, batch_idx % len(train_loader), len(train_loader), loss_D.item(), loss_G.item()))    
            else:
                loss = 1.- pixel_criterion(output,target)
                optimizerG.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(netG.parameters(),opt.clip) 
                optimizerG.step()
                if batch_idx%30 == 0:
                    print("===> Epoch[{}]({}/{}): SSIM Loss: {:.5}".format(epoch, batch_idx, len(train_loader), loss.item()))
        if not os.path.exists('./result/{}_{}/{}_{}x{}'.format(opt.loss_type, opt.code, opt.model, opt.feature_type,opt.rescale_factor)):
            os.makedirs('./result/{}_{}/{}_{}x{}'.format(opt.loss_type, opt.code, opt.model, opt.feature_type,opt.rescale_factor))

        ### valid ###
        PSNR_PREDICTED = 0
        psnr_BICUBIC = 0
        for batch_idx, (lr_f, hr_f) in enumerate(test_loader):
            fig_size = int(math.sqrt(opt.batchSize))
            lr_f = lr_f.cuda()
            sr_f = torch.clamp(netG(lr_f),0,1)
            sr_f = torch.reshape(sr_f,shape)*255

            for i in range(0,fig_size):
                low = sr_f[fig_size*i,:,:,:]
                for j in range(1,fig_size):
                    low = torch.cat((low,sr_f[fig_size*i+j,:,:,:]),0)
                if(i == 0):
                    pic = low
                else:
                    pic = torch.cat((pic,low),1)
            sr_f = pic.cpu().detach().numpy() ###256x1x50x68
            sr_f = sr_f.astype(np.uint8)

            lr_f = torch.reshape(lr_f, (-1,int(shape[1]/opt.rescale_factor),int(shape[2]/opt.rescale_factor),1))*255
            for i in range(0,fig_size):
                low = lr_f[fig_size*i,:,:,:]
                for j in range(1,fig_size):
                    low = torch.cat((low,lr_f[fig_size*i+j,:,:,:]),0)
                if(i == 0):
                    pic = low
                else:
                    pic = torch.cat((pic,low),1)
            lr_f = pic.cpu().detach().numpy() ###256x1x50x68
            lr_f = lr_f.astype(np.uint8)

            hr_f = torch.reshape(hr_f,shape)*255
            for i in range(0,fig_size):
                low = hr_f[fig_size*i,:,:,:]
                for j in range(1,fig_size):
                    low = torch.cat((low,hr_f[fig_size*i+j,:,:,:]),0)
                if(i == 0):
                    pic = low
                else:
                    pic = torch.cat((pic,low),1)
            hr_f = pic.cpu().detach().numpy() ###256x1x50x68
            hr_f = hr_f.astype(np.uint8)
            if(batch_idx<10):
                if(epoch == 1):
                    cv2.imwrite("./result/{}_{}/{}_{}x{}/LR_{}.png".format(opt.loss_type, opt.code, opt.model, opt.feature_type,opt.rescale_factor,batch_idx),lr_f)
                    cv2.imwrite("./result/{}_{}/{}_{}x{}/HR_{}.png".format(opt.loss_type, opt.code, opt.model, opt.feature_type,opt.rescale_factor,batch_idx),hr_f)
                cv2.imwrite("./result/{}_{}/{}_{}x{}/SR_{}.png".format(opt.loss_type, opt.code, opt.model, opt.feature_type,opt.rescale_factor,batch_idx),sr_f)
            PSNR_PREDICTED += utils.psnr(hr_f.astype(float),sr_f.astype(float))

        print("Evaluation ===> Epoch[{}] : PSNR_predicted : {:.5f}".format(epoch, PSNR_PREDICTED/len(test_loader)))
        if(PSNR_PREDICTED/len(test_loader)>best_acc):
            best_acc = PSNR_PREDICTED/len(test_loader)
            save_checkpoint(netG, epoch)
        
        model_out_latest_path = './result/{}_{}/{}_{}x{}/model_latest.pth'.format(opt.loss_type, opt.code, opt.model, opt.feature_type,opt.rescale_factor)
        torch.save(netG.state_dict(), model_out_latest_path)
    
    sys.stdout.close()  



def save_checkpoint(model, epoch):
    print('Saving Networks..')
    model_out_path = './result/{}_{}/{}_{}x{}/model.pth'.format(opt.loss_type, opt.code, opt.model, opt.feature_type,opt.rescale_factor)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    
if __name__ == "__main__":
    main()
