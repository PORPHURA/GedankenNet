############################
#
#  Codes for "Self-supervised learning of hologram reconstruction using physics consistency"
#  Authors: Luzhe Huang, Hanlong Chen, Tairan Liu and Aydogan Ozcan
#  Contact: ozcan@ucla.edu
#
#  test.py: to test a trained GedankenNet model on experimental holograms
#
############################

# %% init
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
from networks.fno import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim

from timeit import default_timer
# from torchinfo import summary
import itertools
import time

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

z_list = [300, 375]  # SPECIFY THE TESTING SAMPLE-TO-SENSOR DISTANCES
z_idx = [0, 1]  # THIS INDEX IS FOR EXPERIMENTAL DATA

# dynamic range for output images
# LUNG
AMP_RG = [0, 1.8]
PH_RG = [-0.6, 1.0]
# PAP SMEAR
# AMP_RG = [0, 1.5]
# PH_RG = [-0.6, 1.2]
# PROSTATE
# AMP_RG = [0, 2.0]
# PH_RG = [-0.4, 0.4]

################################################################
# configs
################################################################

modes = 256
width = 48

batch_size = 1

sub = 1
S = 512 # final resolution


path = 'GedankenNet_ep=10000_m=256_w=48_z=[300, 375]'  # NAME OF THE MODEL
model_name = 'ckpt.pth'  # CHECKPOINT NAME
path_model = 'Models/'+path
path_output = 'outputs/'+path

if not os.path.exists(path_output):
    os.makedirs(path_output)

################################################################
# load data
################################################################

TEST_PATH = ''  # PATH OF EXPERIMENTAL DATA

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

# model = FNO2d(modes, width, in_dim=len(z_list), out_dim=2).cuda()

# print(count_params(model))

# summary(model, input_size=(batch_size, S, S, T_in*T_in_comp))

model = torch.load(os.path.join(path_model, model_name))
model.eval()

with torch.no_grad():
    for i in range(len(test_dataset)):
        loss = 0
        xx, yy = test_dataset[i]
        xx, yy = xx.unsqueeze(axis=0), yy.unsqueeze(axis=0)
        xx = xx.to(device)
        yy = yy.to(device)

        im = model(xx)
        xx, im, yy = xx.cpu().numpy(), im.cpu().numpy().squeeze(), yy.cpu().numpy().squeeze()
        (im_amp, im_ph), (yy_amp, yy_ph) = R2P(comp_field_norm(im)), R2P(comp_field_norm(yy))
        if AMP_RG is not None and PH_RG is not None:
            im_amp = min_max_norm(im_amp, vmin=AMP_RG[0], vmax=AMP_RG[1])
            yy_amp = min_max_norm(yy_amp, vmin=AMP_RG[0], vmax=AMP_RG[1])
            im_ph = min_max_norm(im_ph, vmin=PH_RG[0], vmax=PH_RG[1])
            yy_ph = min_max_norm(yy_ph, vmin=PH_RG[0], vmax=PH_RG[1])
        else:
            im_amp, im_ph = min_max_norm(im_amp, vmin=yy_amp.min(), vmax=yy_amp.max()), min_max_norm(im_ph, vmin=yy_ph.min(), vmax=yy_ph.max())
            yy_amp, yy_ph = min_max_norm(yy_amp), min_max_norm(yy_ph)
        fname = os.path.basename(test_dataset.gt_images[i])
        for i in range(len(z_list)):
            inp = min_max_norm(xx[0,i,...])
            plt.imsave(os.path.join(path_output, fname.replace('.mat','_input%d.jpg'%i)), inp, cmap='gray')
        plt.imsave(os.path.join(path_output, fname.replace('.mat','_target_amp.jpg')), yy_amp, cmap='gray')
        plt.imsave(os.path.join(path_output, fname.replace('.mat','_target_ph.jpg')), yy_ph, cmap='viridis')
        plt.imsave(os.path.join(path_output, fname.replace('.mat','_output_amp.jpg')), im_amp, cmap='gray')
        plt.imsave(os.path.join(path_output, fname.replace('.mat','_output_ph.jpg')), im_ph, cmap='viridis')
        sio.savemat(os.path.join(path_output, fname), {'inputData':xx, 'outputData':im, 'targetData':yy})
