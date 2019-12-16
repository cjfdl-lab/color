# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
# For conversion
from skimage.color import lab2rgb
from skimage import io
# For everything
import torch
import torch.nn as nn
# For our model
import torchvision.models as models
from torchvision import transforms
import os
# For utilities

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Net(nn.Module):
    def __init__(self,root):
        """
        root:保存好的模型的pth文件的路径
        """
        super(Net, self).__init__()
        #改变图片大小，设置tensor张量的类型
        self.transform = transforms.Compose([transforms.Resize((224,224))])
        self.dtype=torch.FloatTensor
        #网络结构
        resnet = models.resnet34() 
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:8])

        self.upsample = nn.Sequential(
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), #14*14
            nn.Conv2d(512,256,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )
        #激活模型
        self.dic=torch.load(root,map_location="cpu")
        self.load_state_dict(self.dic["state_dict"])

    def forward(self,img_root):
        """
        img_root:灰度图片对应的路径
        
        return : 返回一个RGB图像对应的三维数组，可直接调用skimage.io.imsave()存储起来
        """
        self.eval()
        img=Image.open(img_root)
        img_original = self.transform(img)
        img_original = np.asarray(img_original)/255 #归一化
        img_original = torch.from_numpy(img_original).unsqueeze(0).type(self.dtype) #1*224*224
        x = img_original.unsqueeze(0) #模型的输入 1*1*224*224
        midlevel_features = self.midlevel_resnet(x)
        y = self.upsample(midlevel_features).squeeze() #模型的输出 2*224*224
        #将两个tensor对象合起来变成3*224*224
        color_image = torch.cat((img_original,y.detach().cpu()), 0).numpy()
        color_image = color_image.transpose((1, 2, 0))
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
        color_image = lab2rgb(color_image.astype(np.float64))
        return color_image


if __name__=="__main__":
    root="./new_model.pth"
    model=Net(root).to(DEVICE)
