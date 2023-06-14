############################
#
#  Codes for "Self-supervised learning of hologram reconstruction using physics consistency"
#  Authors: Luzhe Huang, Hanlong Chen, Tairan Liu and Aydogan Ozcan
#  Contact: ozcan@ucla.edu
#
#  testP.py: to test a trained GedankenNet-Phase model on experimental data
#
############################


# %% init
import os
# SPECIFY THE GPU ID HERE
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # SPECIFY YOUR GPU ID

import torch
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities import *
import glob

import operator
from functools import reduce
from functools import partial
import np_transforms

from my_tools import *
from networks.unet_parts import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim

from timeit import default_timer
# from torchinfo import summary
import itertools
import time

from Adam import Adam
from networks.fno import FNO2d

torch.manual_seed(0)
np.random.seed(0)


# OBJECT PHASE RANGE MUST BE [0, PI]

z_list = [300, 375]  # SPECIFY THE TESTING SAMPLE-TO-SENSOR DISTANCES, WITHIN RANGE [275, 400], [um]
z_idx = (np.array(z_list) - 275) // 25  # THIS INDEX IS FOR EXPERIMENTAL DATA


################################################################
# configs
################################################################

modes = 256
width = 48

batch_size = 1
batch_size2 = batch_size

sub = 1
S = 512 # final resolution


path = 'GNv2_ep=10000_m=256_w=4_z=[275, 400]_GedankenP1.0'  # MODEL NAME
model_name = 'ckpt.pth'  # CHECKPOINT NAME
path_model = 'Models/'+path
path_output = 'outputs/'+path

if not os.path.exists(path_output):
    os.makedirs(path_output)

################################################################
# load data
################################################################

TEST_PATH = ''  # SPECIFY THE EXPERIMENTAL TESTING DATA PATH

test_file_paths = glob.glob(os.path.join(TEST_PATH, '*demo*.mat'))  # CHANGE THE FORMAT OF YOUR VALID DATA HERE
test_dataset = HoloDataset(test_file_paths, z_idx, np_transforms.Compose([np_transforms.CenterCrop(S),
                                                                        np_transforms.ToTensor()
                                                                ])
            )
print(test_file_paths)

print(len(test_file_paths))


################################################################
# Testing
################################################################

# model = FNO2d(modes, width).cuda()

# print(count_params(model))

# summary(model, input_size=(batch_size, S, S, T_in*T_in_comp))

model = torch.load(os.path.join(path_model, model_name))
model.eval()

with torch.no_grad():
    for i in range(len(test_dataset)):
        loss = 0
        xx, yy = test_dataset[i]
        xx, yy = xx.unsqueeze(axis=0), yy.unsqueeze(axis=0)
        # normalize the input holograms
        xx /= torch.mean(xx, dim=(2,3), keepdim=True)
        xx = xx.to(device)
        yy = yy.to(device)
        # get phase
        yy = torch.angle(yy[:,0:1,...] + 1j*yy[:,1:2,...]) / np.pi

        im, _ = model(xx)
        xx, im, yy = xx.cpu().numpy(), im.cpu().numpy().squeeze(), yy.cpu().numpy().squeeze()
        
        # normalization by substracting phase mean
        im_ph, yy_ph = im - im.mean(axis=(-2,-1), keepdims=True), yy - yy.mean(axis=(-2,-1), keepdims=True)
        im_ph = min_max_norm(im_ph, vmin=np.percentile(yy_ph, 0.25), vmax=np.percentile(yy_ph, 99.95))
        yy_ph = min_max_norm(yy_ph, vmin=np.percentile(yy_ph, 0.25), vmax=np.percentile(yy_ph, 99.95))

        fname = test_dataset.gt_images[i].split('\\')[-1]
        for i in range(len(z_list)):
            inp = min_max_norm(xx[0,i,...])
            plt.imsave(os.path.join(path_output, fname.replace('.mat','_input%d.jpg'%i)), inp, cmap='gray')
        plt.imsave(os.path.join(path_output, fname.replace('.mat','_target_ph.jpg')), yy_ph, cmap='viridis')
        plt.imsave(os.path.join(path_output, fname.replace('.mat','_output_ph.jpg')), im_ph, cmap='viridis')
        sio.savemat(os.path.join(path_output, fname), {'inputData':xx, 'outputData':im, 'targetData':yy})
