import torch
import torch.nn as nn
import numpy as np
from .ops import CustomConv2d_,UpsampleBlock

class Generator(nn.Module):
    def __init__(self,feature_type,scale,multi_scale=False,group=1):
        super(Generator, self).__init__()
        self.channels = [1,4,8,16,32,64]
        self.kernel_size = [5,3,3,3,3]
        self.stride = [1,1,1,1,1,2]
        self.fin = 128 * 48 * 64
        self.scale = scale
        layers = [CustomConv2d_(in_channel,out_channel, k_size, stride) for in_channel, out_channel, k_size, stride in zip(self.channels, self.channels[1:],self.kernel_size,self.stride)]

        if(feature_type == 'p2'):
            ##fin = 128 * 48 * 64 (fixed to 1/4 scale)
            self.fout = [128,96]
        elif(feature_type == 'p3'):
            self.fin = int(self.fin /4)
            self.fout = [64,48]
        elif(feature_type == 'p4'):
            self.fin = int(self.fin /16)
            self.fout = [32,24]        
        elif(feature_type == 'p5'):
            self.fin = int(self.fin /64)
            self.fout = [16,12]
        elif(feature_type == 'p6'):
            self.fin = int(self.fin /256)
            self.fout = [8,6]
        self.upsample = nn.Upsample(scale_factor=scale, mode='bicubic')
        self.blocks = nn.Sequential(*layers)
        self.upscale = UpsampleBlock(64, scale=scale, multi_scale=multi_scale,group=group)

        # self.dropout = nn.Dropout(0.3,inplace=True)
        self.relu = nn.LeakyReLU()
        # self.fc = nn.Linear(self.fin,self.fout[0]*self.fout[1])
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, input):
        output = self.relu(self.blocks(input))
        # output = self.dropout(output)
        # output = torch.add(output,input)
        output = self.upscale(output,scale=self.scale)
        # print(output.size())
        output = self.conv_output(output)
        # output = output.view(output.size(0),-1)
        # output = self.relu(self.fc(output))
        output = output + self.upsample(input)
        return output

class Discriminator(nn.Module):
    def __init__(self,feature_type):
        super(Discriminator, self).__init__()
        self.fout = 32 * 24 * 64
        if(feature_type == 'p3'):
            self.fout = int(self.fout /4)
        elif(feature_type == 'p4'):
            self.fout = int(self.fout /16)
        elif(feature_type == 'p5'):
            self.fout = int(self.fout /64)
        elif(feature_type == 'p6'):
            self.fout = int(self.fout /256)

        self.channels = [1,8,16,32,64]
        self.kernel_size = [5,5,3,3]
        self.stride = [2,2,2,1]
        layers = [CustomConv2d_(in_channel,out_channel, k_size, stride) for in_channel, out_channel, k_size, stride in zip(self.channels, self.channels[1:],self.kernel_size,self.stride)]
        self.blocks = nn.Sequential(*layers)
        self.linear = nn.Linear(self.fout, 1)

    def forward(self, input):
        output = self.blocks(input)
        # print(output.size())
        output = output.view(output.size(0),-1)
        output = self.linear(output)

        return output
