'''Extract feature with VGG16 and Train Feature-SR PyTorch.'''
import os
import sys
import random
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import utils.util as util
import numpy as np
from PIL import Image
from dataset import get_infer_dataloader
from torch.utils.data import TensorDataset, DataLoader
from models.srresnet import _NetG
from models.vdsr import VDSR
from models.carn import CARN
from models.edsr import EDSR
from torch.autograd import Variable 
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model, build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T

from torch.nn.utils.rnn import pad_sequence
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode

parser = argparse.ArgumentParser(description='Inference code')
parser.add_argument("--num_workers",type=int, default=10, help="number of workers")
parser.add_argument("--data_path", type=str, default = "/home/ubuntu/JH/exp1/dataset")
parser.add_argument("--rescale_factor", type=int, default=4, help="rescale factor for using in training")
parser.add_argument("--model_name", type=str,choices= ["VDSR", "CARN", "SRRN","FRGAN","BICUBIC","EDSR"], default='CARN', help="Feature type for usingin training")
parser.add_argument("--loss_type", type=str, choices= ["MSE", "L1", "SmoothL1","vgg_loss","ssim_loss","adv_loss","lpips","BICUBIC"], default='ssim_loss', help="loss type in training")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument("--group", type=int, default=1)
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()
print(opt)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
feature_types = ['p2','p3','p4','p5','p6']
out_pths = []

for feature in feature_types:
    ###DataLoader###
    assert opt.rescale_factor%2==0, 'only for 2,4,8,16,32'
    dataloader = get_infer_dataloader(opt.data_path,feature,opt.rescale_factor,opt.batch_size,opt.num_workers)

    if(feature == 'p2'):
        shape = (-1,192,256,1)
    elif(feature == 'p3'):
        shape = (-1,96,128,1)
    elif(feature == 'p4'):
        shape = (-1,48,64,1)
    elif(feature == 'p5'):
        shape = (-1,24,32,1)
    elif(feature == 'p6'):
        shape = (-1,12,16,1)

    # ###load model###
    if opt.model_name == 'VDSR':
        netG = vdsr(opt.rescale_factor).cuda()
        model_path = './result/{}/VDSR_p2x{}/model.pth'.format(opt.loss_type,opt.rescale_factor)
        checkpoint = torch.load(model_path)
        netG.load_state_dict(checkpoint,strict = True)

    elif opt.model_name == 'SRRN':
        netG = _NetG().cuda()
        model_path = './models/checkpoint/SRRN_'+str(feature)+'x'+str(opt.rescale_factor)+'.pth'
        checkpoint = torch.load(model_path)
        netG.load_state_dict(checkpoint["model"].state_dict(),strict = True)
    elif opt.model_name == 'EDSR':
        netG = EDSR(scale=opt.rescale_factor).cuda()
        model_path = './result/{}/EDSR_p2x{}/model.pth'.format(opt.loss_type,opt.rescale_factor)
        checkpoint = torch.load(model_path)
        netG.load_state_dict(checkpoint,strict = True)

    elif opt.model_name == 'CARN':
        netG = CARN(scale=opt.rescale_factor,group = opt.group).cuda()
        # model_path = './result/{}/CARN_{}x{}/model.pth'.format(opt.loss_type,feature, opt.rescale_factor)
        model_path = './result/{}/CARN_p2x{}/model.pth'.format(opt.loss_type,opt.rescale_factor)
        checkpoint = torch.load(model_path)
        netG.load_state_dict(checkpoint,strict = True)

    elif opt.model_name == 'BICUBIC':
        netG = nn.Upsample(scale_factor=opt.rescale_factor, mode='bicubic')
    
    netG.eval()
    out_pth = "./result/usep2pre/{}/inference/{}_{}x{}".format(opt.loss_type,opt.model_name,feature,opt.rescale_factor)
    out_pths.append(out_pth)
    if not os.path.exists(out_pth):
        os.makedirs(out_pth)
    
    bicu = nn.Upsample(scale_factor=opt.rescale_factor,mode='bicubic').cuda()

    for batch_idx, (lr_f, hr_f) in enumerate(dataloader):
        lr_f = lr_f.cuda()
        sr_f = torch.clamp(netG(lr_f),0,1)
        sr_f = torch.reshape(sr_f,shape)*255
        sr_f = sr_f.cpu().detach().numpy() ###256x1x50x68
        sr_f = sr_f.astype(np.uint8)        

        hr_f = hr_f.cuda()
        hr_f = torch.clamp(hr_f,0,1)
        hr_f = torch.reshape(hr_f,shape)*255
        hr_f = hr_f.cpu().detach().numpy()
        hr_f = hr_f.astype(np.uint8)
        
        for i in range(0,2):
          low = hr_f[2*i,:,:,:]
          low = np.concatenate((low,hr_f[2*i+1,:,:,:]),0)
          if(i == 0):
            pic = low
          else:
            pic = np.concatenate((pic,low),1)
        cv2.imwrite(os.path.join(out_pth,"HR_{}.png".format(int(batch_idx))),pic)

        for i in range(0,2):
          low = sr_f[2*i,:,:,:]
          low = np.concatenate((low,sr_f[2*i+1,:,:,:]),0)
          if(i == 0):
            pic = low
          else:
            pic = np.concatenate((pic,low),1)
        cv2.imwrite(os.path.join(out_pth,"SR_{}.png".format(int(batch_idx))),pic)

        sr_f = torch.clamp(bicu(lr_f),0,1)
        sr_f = torch.reshape(sr_f,shape)*255
        sr_f = sr_f.cpu().detach().numpy() ###256x1x50x68
        sr_f = sr_f.astype(np.uint8)  

        for i in range(0,2):
          low = sr_f[2*i,:,:,:]
          low = np.concatenate((low,sr_f[2*i+1,:,:,:]),0)
          if(i == 0):
            pic = low
          else:
            pic = np.concatenate((pic,low),1)
        cv2.imwrite(os.path.join(out_pth,"LR_{}.png".format(int(batch_idx))),pic)

    print("Save to {}".format(out_pth))

############ evaluation code ############

def myRound(x): # 양수와 음수에 대해 0을 대칭으로 rounding
  abs_x = abs(x)
  val = np.int16(abs_x + 0.5)
  val2 = np.choose(
      x < 0,
      [
        val, val*(-1)
      ]
  )
  return val2

def myClip(x, maxV):
  val = np.choose(
      x > maxV,
      [
        x, maxV
      ]
  )
  return val

image_idx = 0
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()

image_idx = 0
anns = 0

# Original_8bit

image_files = ['001000', '002153', '008021', '009769', '009891', '015335', '017627', '018150', '018837', '022589']
image_files.extend(['022935', '023230', '024610', '025560', '025593', '027620', '155341', '161397', '165336', '166287'])
image_files.extend(['166642', '169996', '172330', '172648', '176606', '176701', '179765', '180101', '186296', '250758'])
image_files.extend(['259382', '267191', '287545', '287649', '289741', '293245', '308328', '309452', '335529', '337987'])
image_files.extend(['338625', '344029', '350122', '389933', '393226', '395343', '395633', '401862', '402473', '402992'])
image_files.extend(['404568', '406997', '408112', '410650', '414385', '414795', '415194', '415536', '416104', '416758'])
image_files.extend(['427055', '428562', '430073', '433204', '447200', '447313', '448448', '452321', '453001', '458755'])
image_files.extend(['462904', '463522', '464089', '468965', '469192', '469246', '471450', '474078', '474881', '475678'])
image_files.extend(['475779', '537802', '542625', '543043', '543300', '543528', '547502', '550691', '553669', '567740'])
image_files.extend(['570688', '570834', '571943', '573391', '574315', '575372', '575970', '578093', '579158', '581100'])


for iter in range(0, 100):
  image_file_number = image_files[image_idx]
  aug = T.ResizeShortestEdge(
    [768,768]
  )

  image = cv2.imread('./dataset/validset_100/000000'+ image_file_number +'.jpg')
  height, width = image.shape[:2]
  image = aug.get_transform(image).apply_image(image)
  image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
  inputs = [{"image": image, "height": height, "width": width}]
  with torch.no_grad():
      images = model.preprocess_image(inputs)  # don't forget to preprocess
      features = model.backbone(images.tensor)  # set of cnn features

  p2_feature_original = features['p2'].to("cpu")
  p3_feature_original = features['p3'].to("cpu")
  p4_feature_original = features['p4'].to("cpu")
  p5_feature_original = features['p5'].to("cpu")
  p6_feature_original = features['p6'].to("cpu")
  
  bitDepth = 8
  maxRange = [0, 0, 0, 0, 0]

  def maxVal(x):
    return pow(2, x)
  def offsetVal(x):
    return pow(2, x-1)

  def maxRange_layer(x):
    absolute_arr = torch.abs(x) * 2
    max_arr = torch.max(absolute_arr)
    return torch.ceil(max_arr)


  act2 = p2_feature_original.squeeze()
  maxRange[0] = maxRange_layer(act2)

  act3 = p3_feature_original.squeeze()
  maxRange[1] = maxRange_layer(act3)

  act4 = p4_feature_original.squeeze()
  maxRange[2] = maxRange_layer(act4)

  act5 = p5_feature_original.squeeze()
  maxRange[3] = maxRange_layer(act5)

  act6 = p6_feature_original.squeeze()
  maxRange[4] = maxRange_layer(act6)

  globals()['maxRange_{}'.format(image_file_number)] = maxRange

  p2_feature_img = Image.open(os.path.join(out_pths[0],'SR_{}.png'.format(iter)))
  p2_feature_arr = np.array(p2_feature_img)
  p2_feature_arr_round = myRound(p2_feature_arr)

  p3_feature_img = Image.open(os.path.join(out_pths[1],'SR_{}.png'.format(iter)))
  p3_feature_arr = np.array(p3_feature_img)
  p3_feature_arr_round = myRound(p3_feature_arr)
  
  p4_feature_img = Image.open(os.path.join(out_pths[2],'SR_{}.png'.format(iter)))
  p4_feature_arr = np.array(p4_feature_img)
  p4_feature_arr_round = myRound(p4_feature_arr)

  p5_feature_img = Image.open(os.path.join(out_pths[3],'SR_{}.png'.format(iter)))
  p5_feature_arr = np.array(p5_feature_img)
  p5_feature_arr_round = myRound(p5_feature_arr)

  p6_feature_img = Image.open(os.path.join(out_pths[4],'SR_{}.png'.format(iter)))
  p6_feature_arr = np.array(p6_feature_img)
  p6_feature_arr_round = myRound(p6_feature_arr)

  # 복원
  recon_p2 = (((p2_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[0].numpy())
  recon_p3 = (((p3_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[1].numpy())
  recon_p4 = (((p4_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[2].numpy())
  recon_p5 = (((p5_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[3].numpy())
  recon_p6 = (((p6_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[4].numpy())

  # tensor_value = recon_p2
  tensor_value = torch.as_tensor(recon_p2.astype("float32"))
  tensor_value2 = torch.as_tensor(recon_p3.astype("float32"))
  tensor_value3 = torch.as_tensor(recon_p4.astype("float32"))
  tensor_value4 = torch.as_tensor(recon_p5.astype("float32"))
  tensor_value5 = torch.as_tensor(recon_p6.astype("float32"))

  # # MSB 코드 끝

  # lsb 및 원래 코드
  # 복원
  # recon_p2 = (((p2_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[0].numpy())
  # recon_p3 = (((p3_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[1].numpy())
  # recon_p4 = (((p4_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[2].numpy())
  # recon_p5 = (((p5_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[3].numpy())
  # recon_p6 = (((p6_feature_arr_round - offsetVal(bitDepth)) / maxVal(bitDepth)) * maxRange[4].numpy())

  t = [None] * 16
  t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15] = torch.chunk(tensor_value, 16, dim=0)
  p2 = [None] * 256

  t2 = [None] * 16
  t2[0], t2[1], t2[2], t2[3], t2[4], t2[5], t2[6], t2[7], t2[8], t2[9], t2[10], t2[11], t2[12], t2[13], t2[14], t2[15] = torch.chunk(tensor_value2, 16, dim=0)
  p3 = [None] * 256

  t3 = [None] * 16
  t3[0], t3[1], t3[2], t3[3], t3[4], t3[5], t3[6], t3[7], t3[8], t3[9], t3[10], t3[11], t3[12], t3[13], t3[14], t3[15] = torch.chunk(tensor_value3, 16, dim=0)
  p4 = [None] * 256

  t4 = [None] * 16
  t4[0], t4[1], t4[2], t4[3], t4[4], t4[5], t4[6], t4[7], t4[8], t4[9], t4[10], t4[11], t4[12], t4[13], t4[14], t4[15] = torch.chunk(tensor_value4, 16, dim=0)
  p5 = [None] * 256

  t5 = [None] * 16
  t5[0], t5[1], t5[2], t5[3], t5[4], t5[5], t5[6], t5[7], t5[8], t5[9], t5[10], t5[11], t5[12], t5[13], t5[14], t5[15] = torch.chunk(tensor_value5, 16, dim=0)
  p6 = [None] * 256

  p2[0], p2[1], p2[2], p2[3], p2[4], p2[5], p2[6], p2[7], p2[8], p2[9], p2[10], p2[11], p2[12], p2[13], p2[14], p2[15] = torch.chunk(t[0], 16, dim=1)
  p2[16], p2[17], p2[18], p2[19], p2[20], p2[21], p2[22], p2[23], p2[24], p2[25], p2[26], p2[27], p2[28], p2[29], p2[30], p2[31] = torch.chunk(t[1], 16, dim=1)
  p2[32], p2[33], p2[34], p2[35], p2[36], p2[37], p2[38], p2[39], p2[40], p2[41], p2[42], p2[43], p2[44], p2[45], p2[46], p2[47] = torch.chunk(t[2], 16, dim=1)
  p2[48], p2[49], p2[50], p2[51], p2[52], p2[53], p2[54], p2[55], p2[56], p2[57], p2[58], p2[59], p2[60], p2[61], p2[62], p2[63] = torch.chunk(t[3], 16, dim=1)
  p2[64], p2[65], p2[66], p2[67], p2[68], p2[69], p2[70], p2[71], p2[72], p2[73], p2[74], p2[75], p2[76], p2[77], p2[78], p2[79] = torch.chunk(t[4], 16, dim=1)
  p2[80], p2[81], p2[82], p2[83], p2[84], p2[85], p2[86], p2[87], p2[88], p2[89], p2[90], p2[91], p2[92], p2[93], p2[94], p2[95] = torch.chunk(t[5], 16, dim=1)
  p2[96], p2[97], p2[98], p2[99], p2[100], p2[101], p2[102], p2[103], p2[104], p2[105], p2[106], p2[107], p2[108], p2[109], p2[110], p2[111] = torch.chunk(t[6], 16, dim=1)
  p2[112], p2[113], p2[114], p2[115], p2[116], p2[117], p2[118], p2[119], p2[120], p2[121], p2[122], p2[123], p2[124], p2[125], p2[126], p2[127] = torch.chunk(t[7], 16, dim=1)
  p2[128], p2[129], p2[130], p2[131], p2[132], p2[133], p2[134], p2[135], p2[136], p2[137], p2[138], p2[139], p2[140], p2[141], p2[142], p2[143] = torch.chunk(t[8], 16, dim=1)
  p2[144], p2[145], p2[146], p2[147], p2[148], p2[149], p2[150], p2[151], p2[152], p2[153], p2[154], p2[155], p2[156], p2[157], p2[158], p2[159] = torch.chunk(t[9], 16, dim=1)
  p2[160], p2[161], p2[162], p2[163], p2[164], p2[165], p2[166], p2[167], p2[168], p2[169], p2[170], p2[171], p2[172], p2[173], p2[174], p2[175] = torch.chunk(t[10], 16, dim=1)
  p2[176], p2[177], p2[178], p2[179], p2[180], p2[181], p2[182], p2[183], p2[184], p2[185], p2[186], p2[187], p2[188], p2[189], p2[190], p2[191] = torch.chunk(t[11], 16, dim=1)
  p2[192], p2[193], p2[194], p2[195], p2[196], p2[197], p2[198], p2[199], p2[200], p2[201], p2[202], p2[203], p2[204], p2[205], p2[206], p2[207] = torch.chunk(t[12], 16, dim=1)
  p2[208], p2[209], p2[210], p2[211], p2[212], p2[213], p2[214], p2[215], p2[216], p2[217], p2[218], p2[219], p2[220], p2[221], p2[222], p2[223] = torch.chunk(t[13], 16, dim=1)
  p2[224], p2[225], p2[226], p2[227], p2[228], p2[229], p2[230], p2[231], p2[232], p2[233], p2[234], p2[235], p2[236], p2[237], p2[238], p2[239] = torch.chunk(t[14], 16, dim=1)
  p2[240], p2[241], p2[242], p2[243], p2[244], p2[245], p2[246], p2[247], p2[248], p2[249], p2[250], p2[251], p2[252], p2[253], p2[254], p2[255] = torch.chunk(t[15], 16, dim=1)

  p3[0], p3[1], p3[2], p3[3], p3[4], p3[5], p3[6], p3[7], p3[8], p3[9], p3[10], p3[11], p3[12], p3[13], p3[14], p3[15] = torch.chunk(t2[0], 16, dim=1)
  p3[16], p3[17], p3[18], p3[19], p3[20], p3[21], p3[22], p3[23], p3[24], p3[25], p3[26], p3[27], p3[28], p3[29], p3[30], p3[31] = torch.chunk(t2[1], 16, dim=1)
  p3[32], p3[33], p3[34], p3[35], p3[36], p3[37], p3[38], p3[39], p3[40], p3[41], p3[42], p3[43], p3[44], p3[45], p3[46], p3[47] = torch.chunk(t2[2], 16, dim=1)
  p3[48], p3[49], p3[50], p3[51], p3[52], p3[53], p3[54], p3[55], p3[56], p3[57], p3[58], p3[59], p3[60], p3[61], p3[62], p3[63] = torch.chunk(t2[3], 16, dim=1)
  p3[64], p3[65], p3[66], p3[67], p3[68], p3[69], p3[70], p3[71], p3[72], p3[73], p3[74], p3[75], p3[76], p3[77], p3[78], p3[79] = torch.chunk(t2[4], 16, dim=1)
  p3[80], p3[81], p3[82], p3[83], p3[84], p3[85], p3[86], p3[87], p3[88], p3[89], p3[90], p3[91], p3[92], p3[93], p3[94], p3[95] = torch.chunk(t2[5], 16, dim=1)
  p3[96], p3[97], p3[98], p3[99], p3[100], p3[101], p3[102], p3[103], p3[104], p3[105], p3[106], p3[107], p3[108], p3[109], p3[110], p3[111] = torch.chunk(t2[6], 16, dim=1)
  p3[112], p3[113], p3[114], p3[115], p3[116], p3[117], p3[118], p3[119], p3[120], p3[121], p3[122], p3[123], p3[124], p3[125], p3[126], p3[127] = torch.chunk(t2[7], 16, dim=1)
  p3[128], p3[129], p3[130], p3[131], p3[132], p3[133], p3[134], p3[135], p3[136], p3[137], p3[138], p3[139], p3[140], p3[141], p3[142], p3[143] = torch.chunk(t2[8], 16, dim=1)
  p3[144], p3[145], p3[146], p3[147], p3[148], p3[149], p3[150], p3[151], p3[152], p3[153], p3[154], p3[155], p3[156], p3[157], p3[158], p3[159] = torch.chunk(t2[9], 16, dim=1)
  p3[160], p3[161], p3[162], p3[163], p3[164], p3[165], p3[166], p3[167], p3[168], p3[169], p3[170], p3[171], p3[172], p3[173], p3[174], p3[175] = torch.chunk(t2[10], 16, dim=1)
  p3[176], p3[177], p3[178], p3[179], p3[180], p3[181], p3[182], p3[183], p3[184], p3[185], p3[186], p3[187], p3[188], p3[189], p3[190], p3[191] = torch.chunk(t2[11], 16, dim=1)
  p3[192], p3[193], p3[194], p3[195], p3[196], p3[197], p3[198], p3[199], p3[200], p3[201], p3[202], p3[203], p3[204], p3[205], p3[206], p3[207] = torch.chunk(t2[12], 16, dim=1)
  p3[208], p3[209], p3[210], p3[211], p3[212], p3[213], p3[214], p3[215], p3[216], p3[217], p3[218], p3[219], p3[220], p3[221], p3[222], p3[223] = torch.chunk(t2[13], 16, dim=1)
  p3[224], p3[225], p3[226], p3[227], p3[228], p3[229], p3[230], p3[231], p3[232], p3[233], p3[234], p3[235], p3[236], p3[237], p3[238], p3[239] = torch.chunk(t2[14], 16, dim=1)
  p3[240], p3[241], p3[242], p3[243], p3[244], p3[245], p3[246], p3[247], p3[248], p3[249], p3[250], p3[251], p3[252], p3[253], p3[254], p3[255] = torch.chunk(t2[15], 16, dim=1)

  p4[0], p4[1], p4[2], p4[3], p4[4], p4[5], p4[6], p4[7], p4[8], p4[9], p4[10], p4[11], p4[12], p4[13], p4[14], p4[15] = torch.chunk(t3[0], 16, dim=1)
  p4[16], p4[17], p4[18], p4[19], p4[20], p4[21], p4[22], p4[23], p4[24], p4[25], p4[26], p4[27], p4[28], p4[29], p4[30], p4[31] = torch.chunk(t3[1], 16, dim=1)
  p4[32], p4[33], p4[34], p4[35], p4[36], p4[37], p4[38], p4[39], p4[40], p4[41], p4[42], p4[43], p4[44], p4[45], p4[46], p4[47] = torch.chunk(t3[2], 16, dim=1)
  p4[48], p4[49], p4[50], p4[51], p4[52], p4[53], p4[54], p4[55], p4[56], p4[57], p4[58], p4[59], p4[60], p4[61], p4[62], p4[63] = torch.chunk(t3[3], 16, dim=1)
  p4[64], p4[65], p4[66], p4[67], p4[68], p4[69], p4[70], p4[71], p4[72], p4[73], p4[74], p4[75], p4[76], p4[77], p4[78], p4[79] = torch.chunk(t3[4], 16, dim=1)
  p4[80], p4[81], p4[82], p4[83], p4[84], p4[85], p4[86], p4[87], p4[88], p4[89], p4[90], p4[91], p4[92], p4[93], p4[94], p4[95] = torch.chunk(t3[5], 16, dim=1)
  p4[96], p4[97], p4[98], p4[99], p4[100], p4[101], p4[102], p4[103], p4[104], p4[105], p4[106], p4[107], p4[108], p4[109], p4[110], p4[111] = torch.chunk(t3[6], 16, dim=1)
  p4[112], p4[113], p4[114], p4[115], p4[116], p4[117], p4[118], p4[119], p4[120], p4[121], p4[122], p4[123], p4[124], p4[125], p4[126], p4[127] = torch.chunk(t3[7], 16, dim=1)
  p4[128], p4[129], p4[130], p4[131], p4[132], p4[133], p4[134], p4[135], p4[136], p4[137], p4[138], p4[139], p4[140], p4[141], p4[142], p4[143] = torch.chunk(t3[8], 16, dim=1)
  p4[144], p4[145], p4[146], p4[147], p4[148], p4[149], p4[150], p4[151], p4[152], p4[153], p4[154], p4[155], p4[156], p4[157], p4[158], p4[159] = torch.chunk(t3[9], 16, dim=1)
  p4[160], p4[161], p4[162], p4[163], p4[164], p4[165], p4[166], p4[167], p4[168], p4[169], p4[170], p4[171], p4[172], p4[173], p4[174], p4[175] = torch.chunk(t3[10], 16, dim=1)
  p4[176], p4[177], p4[178], p4[179], p4[180], p4[181], p4[182], p4[183], p4[184], p4[185], p4[186], p4[187], p4[188], p4[189], p4[190], p4[191] = torch.chunk(t3[11], 16, dim=1)
  p4[192], p4[193], p4[194], p4[195], p4[196], p4[197], p4[198], p4[199], p4[200], p4[201], p4[202], p4[203], p4[204], p4[205], p4[206], p4[207] = torch.chunk(t3[12], 16, dim=1)
  p4[208], p4[209], p4[210], p4[211], p4[212], p4[213], p4[214], p4[215], p4[216], p4[217], p4[218], p4[219], p4[220], p4[221], p4[222], p4[223] = torch.chunk(t3[13], 16, dim=1)
  p4[224], p4[225], p4[226], p4[227], p4[228], p4[229], p4[230], p4[231], p4[232], p4[233], p4[234], p4[235], p4[236], p4[237], p4[238], p4[239] = torch.chunk(t3[14], 16, dim=1)
  p4[240], p4[241], p4[242], p4[243], p4[244], p4[245], p4[246], p4[247], p4[248], p4[249], p4[250], p4[251], p4[252], p4[253], p4[254], p4[255] = torch.chunk(t3[15], 16, dim=1)

  p5[0], p5[1], p5[2], p5[3], p5[4], p5[5], p5[6], p5[7], p5[8], p5[9], p5[10], p5[11], p5[12], p5[13], p5[14], p5[15] = torch.chunk(t4[0], 16, dim=1)
  p5[16], p5[17], p5[18], p5[19], p5[20], p5[21], p5[22], p5[23], p5[24], p5[25], p5[26], p5[27], p5[28], p5[29], p5[30], p5[31] = torch.chunk(t4[1], 16, dim=1)
  p5[32], p5[33], p5[34], p5[35], p5[36], p5[37], p5[38], p5[39], p5[40], p5[41], p5[42], p5[43], p5[44], p5[45], p5[46], p5[47] = torch.chunk(t4[2], 16, dim=1)
  p5[48], p5[49], p5[50], p5[51], p5[52], p5[53], p5[54], p5[55], p5[56], p5[57], p5[58], p5[59], p5[60], p5[61], p5[62], p5[63] = torch.chunk(t4[3], 16, dim=1)
  p5[64], p5[65], p5[66], p5[67], p5[68], p5[69], p5[70], p5[71], p5[72], p5[73], p5[74], p5[75], p5[76], p5[77], p5[78], p5[79] = torch.chunk(t4[4], 16, dim=1)
  p5[80], p5[81], p5[82], p5[83], p5[84], p5[85], p5[86], p5[87], p5[88], p5[89], p5[90], p5[91], p5[92], p5[93], p5[94], p5[95] = torch.chunk(t4[5], 16, dim=1)
  p5[96], p5[97], p5[98], p5[99], p5[100], p5[101], p5[102], p5[103], p5[104], p5[105], p5[106], p5[107], p5[108], p5[109], p5[110], p5[111] = torch.chunk(t4[6], 16, dim=1)
  p5[112], p5[113], p5[114], p5[115], p5[116], p5[117], p5[118], p5[119], p5[120], p5[121], p5[122], p5[123], p5[124], p5[125], p5[126], p5[127] = torch.chunk(t4[7], 16, dim=1)
  p5[128], p5[129], p5[130], p5[131], p5[132], p5[133], p5[134], p5[135], p5[136], p5[137], p5[138], p5[139], p5[140], p5[141], p5[142], p5[143] = torch.chunk(t4[8], 16, dim=1)
  p5[144], p5[145], p5[146], p5[147], p5[148], p5[149], p5[150], p5[151], p5[152], p5[153], p5[154], p5[155], p5[156], p5[157], p5[158], p5[159] = torch.chunk(t4[9], 16, dim=1)
  p5[160], p5[161], p5[162], p5[163], p5[164], p5[165], p5[166], p5[167], p5[168], p5[169], p5[170], p5[171], p5[172], p5[173], p5[174], p5[175] = torch.chunk(t4[10], 16, dim=1)
  p5[176], p5[177], p5[178], p5[179], p5[180], p5[181], p5[182], p5[183], p5[184], p5[185], p5[186], p5[187], p5[188], p5[189], p5[190], p5[191] = torch.chunk(t4[11], 16, dim=1)
  p5[192], p5[193], p5[194], p5[195], p5[196], p5[197], p5[198], p5[199], p5[200], p5[201], p5[202], p5[203], p5[204], p5[205], p5[206], p5[207] = torch.chunk(t4[12], 16, dim=1)
  p5[208], p5[209], p5[210], p5[211], p5[212], p5[213], p5[214], p5[215], p5[216], p5[217], p5[218], p5[219], p5[220], p5[221], p5[222], p5[223] = torch.chunk(t4[13], 16, dim=1)
  p5[224], p5[225], p5[226], p5[227], p5[228], p5[229], p5[230], p5[231], p5[232], p5[233], p5[234], p5[235], p5[236], p5[237], p5[238], p5[239] = torch.chunk(t4[14], 16, dim=1)
  p5[240], p5[241], p5[242], p5[243], p5[244], p5[245], p5[246], p5[247], p5[248], p5[249], p5[250], p5[251], p5[252], p5[253], p5[254], p5[255] = torch.chunk(t4[15], 16, dim=1)

  p6[0], p6[1], p6[2], p6[3], p6[4], p6[5], p6[6], p6[7], p6[8], p6[9], p6[10], p6[11], p6[12], p6[13], p6[14], p6[15] = torch.chunk(t5[0], 16, dim=1)
  p6[16], p6[17], p6[18], p6[19], p6[20], p6[21], p6[22], p6[23], p6[24], p6[25], p6[26], p6[27], p6[28], p6[29], p6[30], p6[31] = torch.chunk(t5[1], 16, dim=1)
  p6[32], p6[33], p6[34], p6[35], p6[36], p6[37], p6[38], p6[39], p6[40], p6[41], p6[42], p6[43], p6[44], p6[45], p6[46], p6[47] = torch.chunk(t5[2], 16, dim=1)
  p6[48], p6[49], p6[50], p6[51], p6[52], p6[53], p6[54], p6[55], p6[56], p6[57], p6[58], p6[59], p6[60], p6[61], p6[62], p6[63] = torch.chunk(t5[3], 16, dim=1)
  p6[64], p6[65], p6[66], p6[67], p6[68], p6[69], p6[70], p6[71], p6[72], p6[73], p6[74], p6[75], p6[76], p6[77], p6[78], p6[79] = torch.chunk(t5[4], 16, dim=1)
  p6[80], p6[81], p6[82], p6[83], p6[84], p6[85], p6[86], p6[87], p6[88], p6[89], p6[90], p6[91], p6[92], p6[93], p6[94], p6[95] = torch.chunk(t5[5], 16, dim=1)
  p6[96], p6[97], p6[98], p6[99], p6[100], p6[101], p6[102], p6[103], p6[104], p6[105], p6[106], p6[107], p6[108], p6[109], p6[110], p6[111] = torch.chunk(t5[6], 16, dim=1)
  p6[112], p6[113], p6[114], p6[115], p6[116], p6[117], p6[118], p6[119], p6[120], p6[121], p6[122], p6[123], p6[124], p6[125], p6[126], p6[127] = torch.chunk(t5[7], 16, dim=1)
  p6[128], p6[129], p6[130], p6[131], p6[132], p6[133], p6[134], p6[135], p6[136], p6[137], p6[138], p6[139], p6[140], p6[141], p6[142], p6[143] = torch.chunk(t5[8], 16, dim=1)
  p6[144], p6[145], p6[146], p6[147], p6[148], p6[149], p6[150], p6[151], p6[152], p6[153], p6[154], p6[155], p6[156], p6[157], p6[158], p6[159] = torch.chunk(t5[9], 16, dim=1)
  p6[160], p6[161], p6[162], p6[163], p6[164], p6[165], p6[166], p6[167], p6[168], p6[169], p6[170], p6[171], p6[172], p6[173], p6[174], p6[175] = torch.chunk(t5[10], 16, dim=1)
  p6[176], p6[177], p6[178], p6[179], p6[180], p6[181], p6[182], p6[183], p6[184], p6[185], p6[186], p6[187], p6[188], p6[189], p6[190], p6[191] = torch.chunk(t5[11], 16, dim=1)
  p6[192], p6[193], p6[194], p6[195], p6[196], p6[197], p6[198], p6[199], p6[200], p6[201], p6[202], p6[203], p6[204], p6[205], p6[206], p6[207] = torch.chunk(t5[12], 16, dim=1)
  p6[208], p6[209], p6[210], p6[211], p6[212], p6[213], p6[214], p6[215], p6[216], p6[217], p6[218], p6[219], p6[220], p6[221], p6[222], p6[223] = torch.chunk(t5[13], 16, dim=1)
  p6[224], p6[225], p6[226], p6[227], p6[228], p6[229], p6[230], p6[231], p6[232], p6[233], p6[234], p6[235], p6[236], p6[237], p6[238], p6[239] = torch.chunk(t5[14], 16, dim=1)
  p6[240], p6[241], p6[242], p6[243], p6[244], p6[245], p6[246], p6[247], p6[248], p6[249], p6[250], p6[251], p6[252], p6[253], p6[254], p6[255] = torch.chunk(t5[15], 16, dim=1)

  p2_tensor = pad_sequence(p2, batch_first=True)
  p3_tensor = pad_sequence(p3, batch_first=True)
  p4_tensor = pad_sequence(p4, batch_first=True)
  p5_tensor = pad_sequence(p5, batch_first=True)
  p6_tensor = pad_sequence(p6, batch_first=True)
  

  cc = p2_tensor.unsqueeze(0)
  cc2 = p3_tensor.unsqueeze(0)
  cc3 = p4_tensor.unsqueeze(0)
  cc4 = p5_tensor.unsqueeze(0)
  cc5 = p6_tensor.unsqueeze(0)

  p2_cuda = cc.to(torch.device("cuda"))
  p3_cuda = cc2.to(torch.device("cuda"))
  p4_cuda = cc3.to(torch.device("cuda"))
  p5_cuda = cc4.to(torch.device("cuda"))
  p6_cuda = cc5.to(torch.device("cuda"))
  
  aug = T.ResizeShortestEdge([768,768])
  image = cv2.imread('./dataset/validset_100/000000'+ image_file_number +'.jpg') #(480,640,3)
  height, width = image.shape[:2]
  image = aug.get_transform(image).apply_image(image) #(768,1024,3)
  image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
  inputs = [{"image": image, "height": height, "width": width}]
  with torch.no_grad():
      images = model.preprocess_image(inputs)  # don't forget to preprocess #torch.Size([1, 3, 768, 1024])
      features = model.backbone(images.tensor)  # set of cnn features
      features['p2'] = p2_cuda
      features['p3'] = p3_cuda
      features['p4'] = p4_cuda
      features['p5'] = p5_cuda
      # features['p6'] = p6_cuda

      proposals, _ = model.proposal_generator(images, features, None)  # RPN

      features_ = [features[f] for f in model.roi_heads.box_in_features]
      box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
      box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
      predictions = model.roi_heads.box_predictor(box_features)
      pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
      pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

      # output boxes, masks, scores, etc
      pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
      # features of the proposed boxes
      feats = box_features[pred_inds]

  pred_category = pred_instances[0]["instances"].pred_classes.to("cpu")
  pred_segmentation = pred_instances[0]["instances"].pred_masks.to("cpu")
  pred_score = pred_instances[0]["instances"].scores.to("cpu")

  xxx = pred_category
  xxx = xxx.numpy()

  xxx = xxx + 1

  for idx in range(len(xxx)):
      if -1 < int(xxx[idx]) < 12:
        xxx[idx] = xxx[idx]
      elif 11 < int(xxx[idx]) < 25:
        xxx[idx] = xxx[idx] + 1
      elif 24 < int(xxx[idx]) < 27:
        xxx[idx] = xxx[idx] + 2
      elif 26 < int(xxx[idx]) < 41:
        xxx[idx] = xxx[idx] + 4
      elif 40 < int(xxx[idx]) < 61:
        xxx[idx] = xxx[idx] + 5
      elif 60 < int(xxx[idx]) < 62:
        xxx[idx] = 67
      elif 61 < int(xxx[idx]) < 63:
        xxx[idx] = 70
      elif 62 < int(xxx[idx]) < 74:
        xxx[idx] = xxx[idx] + 9
      else:
        xxx[idx] = xxx[idx] + 10

  imgID = int(image_file_number)
  if image_idx == 0:
    anns = []
  else:
    anns = anns

  for idx in range(len(pred_category.numpy())):

    anndata = {}
    anndata['image_id'] = imgID
    anndata['category_id'] = int(xxx[idx])
      
    anndata['segmentation'] = encode(np.asfortranarray(pred_segmentation[idx].numpy()))
    anndata['score'] = float(pred_score[idx].numpy())
    anns.append(anndata)

  image_idx = image_idx + 1
  # print("###image###:{}".format(image_idx))

annType = ['segm','bbox','keypoints']
annType = annType[0]      #specify type here
prefix = 'instances'
print('Running demo for *%s* results.'%(annType)) 
# imgIds = [560474]
annFile = './instances_val2017_dataset100.json'
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile = anns
cocoDt=cocoGt.loadRes(resFile)

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
# cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()