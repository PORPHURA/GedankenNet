############################
#
#  Codes for "Self-supervised learning of hologram reconstruction using physics consistency"
#  Authors: Luzhe Huang, Hanlong Chen, Tairan Liu and Aydogan Ozcan
#  Contact: ozcan@ucla.edu
#
#  train_Gedanken.py: to train GedankenNet models
#
############################

# %% init
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # SPECIFY YOUR GPU ID

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities import *
import glob
from skimage.metrics import structural_similarity as ssim

import operator
from functools import reduce
from functools import partial

from my_tools import *
from networks.fno import FNO2d
import np_transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.profiler import profile, record_function, ProfilerActivity

from timeit import default_timer
# from torchinfo import summary

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

z_list = [300, 375]  # SPECIFY TRAINING SAMPLE-TO-SENSOR DISTANCES, [um]

from torchvision.models import vgg16_bn


def tv_loss(inputs):
    # inputs: [N, C, H, W]
    n, c, h, w = inputs.shape
    grad_x = inputs[:,:,1:,:] - inputs[:,:,:-1,:]
    grad_y = inputs[:,:,:,1:] - inputs[:,:,:,:-1]
    tv = (grad_x.abs().sum() + grad_y.abs().sum()) / (n*c*h*w)
    return tv


###############################################################
# Physics consistency loss
###############################################################

def batch_fsp(batch, z_list):
    # batch: [N, 2, H, W]
    prop_batch = []
    for n in range(batch.shape[0]):
        comp_field = torch.complex(batch[n,0,...], batch[n,1,...])
        wave = AxialPlane_torch(comp_field, 0, params)
        prop_field = torch.stack([wave(z) for z in z_list], dim=0)  # [C=2, H, W]
        prop_batch.append(prop_field)
    return torch.stack(prop_batch, axis=0)  # [N, C=2, H, W]


class AxialPlane_torch():
    def __init__(self, init_guess, z0, params) -> None:

        # initialize complex field
        self.comp_field = init_guess

        # parameters
        self.z0 = z0
        self.wl = params['wavelength'] / params['ref_ind']  # effective wavelength, [um]
        self.k = 1 / self.wl
        self.n_pixel = params['patch_size']
        self.pixel_size = params['pixel_size']  # [um]
        self.max_freq = 1/self.pixel_size  # [um-1]

    def __call__(self, z):

        # free space propagation using angular spectrum
        ang_spectrum = torch.fft.fftshift(torch.fft.fft2(self.comp_field))

        # create grid                                                    
        uu, vv = np.meshgrid(self.max_freq*np.arange(-np.ceil((self.n_pixel-1)/2), np.floor((self.n_pixel+1)/2))/self.n_pixel,
                    self.max_freq*np.arange(-np.ceil((self.n_pixel-1)/2), np.floor((self.n_pixel+1)/2))/self.n_pixel)
        mask = ((self.k**2 - uu**2 - vv**2) >= 0).astype('float32')
        ww = np.sqrt((self.k**2-uu**2-vv**2)*mask).astype('float32')
        # transfer function
        dz = z - self.z0
        h = np.exp(1j * 2 * np.pi * ww * dz) * mask
        ang_spectrum *= torch.from_numpy(h).to(device)

        # IFFT
        prop_field = torch.fft.ifft2(torch.fft.ifftshift(ang_spectrum))

        return prop_field



################################################################
# configs
################################################################

TRAIN_PATH = ''  # SPECIFY YOUR TRAINING DATA PATH
VALID_PATH = ''  # SPECIFY YOUR VALIDATION DATA PATH

ntrain_files = 99999
nvalid_files = 16
nvalid_files = 16

modes = 256
width = 48

batch_size = 2
batch_per_ep = 250

epochs = 1000
learning_rate = 0.002

scheduler_amma = 0.5
scheduler_step = list(range(85,epochs,20))
# print(epochs, learning_rate, scheduler_step, scheduler_gamma)

sub = 1
S = 512 # final resolution
step = 1

# wavelength: [um], pixel_size: [um], patch_size: [px], ref_ind: refractive index of the medium, air is 1.0
params = {'wavelength':0.530,'pixel_size':0.3733,'patch_size':S,'ref_ind':1.00}  # SPECIFY YOUR TRAINING SETUPS



def main():
    ################################################################
    # load data
    ################################################################
    train_file_paths = glob.glob(os.path.join(TRAIN_PATH, '*.png'))
    train_dataset = GedankenDataset(train_file_paths, z_list, np_transforms.Compose([np_transforms.RandomCrop(S),
                                                                                np_transforms.RandomHorizontalFlip(),
                                                                                np_transforms.ToTensor()
                                                                        ]), params)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    valid_file_paths = glob.glob(os.path.join(VALID_PATH, '*.png'))  # CHANGE THE FORMAT OF YOUR VALID DATA HERE
    valid_dataset = GedankenDataset(valid_file_paths, z_list, np_transforms.Compose([np_transforms.RandomCrop(S),
                                                                                np_transforms.RandomHorizontalFlip(),
                                                                                np_transforms.ToTensor()
                                                                        ]), params)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    path = 'FNO_ep=%d_m=%d_w=%d_z=%s_Gedanken' % (epochs, modes, width, z_list)
    path_model = 'Models/'+path
    path_train_err = 'results/'+path+'train.txt'
    path_valid_err = 'results/'+path+'valid.txt'
    path_image = 'image/'+path

    writer = SummaryWriter(os.path.join("runs", path))


    ################################################################
    # training and evaluation
    ################################################################

    model = FNO2d(modes, width, len(z_list), 2).cuda()
    # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

    print(count_params(model))

    # summary(model, input_size=(batch_size, S, S, T_in*T_in_comp))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

    # myloss = LpLoss(size_average=False)
    maeloss = nn.L1Loss(reduction='mean')
    mseloss = nn.MSELoss()

    # Fourier domain window
    hann_window = torch.outer(torch.hann_window(S), torch.hann_window(S))  # 2D Hann window
    hann_window = torch.fft.ifftshift(hann_window).unsqueeze(0).unsqueeze(0).cuda()

    if not os.path.exists(path_model):
        os.makedirs(path_model)


    print_target = False
    start_ep = -1
    min_valid_mse = 1e4
    if os.path.isfile(os.path.join(path_model,"checkpoint.pth")):
        checkpoint = torch.load(os.path.join(path_model,"checkpoint.pth"), map_location='cpu')
        start_ep = checkpoint['epoch']
        print("Resuming from the checkpoint: ep", start_ep+1)
        np.random.set_state(checkpoint['np_rand_state'])
        torch.set_rng_state(checkpoint['torch_rand_state'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!warning!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("!!!!!!!!!!!!!!!!Temporarily change step size!!!!!!!!!!!!!!!!!")
        # scheduler.step_size = 50
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])

    for ep in range(start_ep+1, epochs):

        print("current learning rate:", optimizer.param_groups[0]['lr'])

        ###################train##############
        model.train()
        t1 = default_timer()
        train_mse = 0

        for i, (xx, yy) in enumerate(train_loader):

            # break when reach batch_per_ep limit
            if i >= batch_per_ep:
                break

            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            im = model(xx)
            im_x = torch.abs(batch_fsp(im, z_list))
            n, t, h, w = xx.shape

            loss += maeloss(torch.fft.fft2(im_x) * hann_window, torch.fft.fft2(xx) * hann_window) * 0.1  # LOSS WEIGHTS MAY VARY FROM CASE TO CASE
            loss += mseloss(im_x, xx)*1.0 + tv_loss(im)*20.0

            train_mse += mseloss(comp_field_norm(im.detach().cpu()), comp_field_norm(yy.detach().cpu())).detach().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            
        ###############valid##################
        valid_mse = 0
        xx_list = []
        yy_list = []
        im_list = []
        with torch.no_grad():
            for i, (xx, yy) in enumerate(valid_loader):
                if i >= 1:
                    break

                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)

                im = model(xx)

                valid_mse += mseloss(comp_field_norm(im.cpu()), comp_field_norm(yy.cpu())).item()
                xx_list.append(xx.cpu().numpy())
                yy_list.append(comp_field_norm(yy).cpu().numpy())
                im_list.append(comp_field_norm(im).cpu().numpy())

        xx = np.vstack(xx_list).reshape((-1,)+xx.shape[1:])
        yy = np.vstack(yy_list).reshape((-1,)+yy.shape[1:])
        im = np.vstack(im_list).reshape((-1,)+im.shape[1:])

        # writer.add_images('input', np.clip(xx,0,255).astype('int'), ep, dataformats='NCHW')
        writer.add_images('output_real', np.clip((im[:,0:1,...]-yy[:,0:1,...].min())/(yy[:,0:1,...].max()-yy[:,0:1,...].min()),0,1), ep, dataformats='NCHW')
        writer.add_images('output_imag', np.clip((im[:,1:2,...]-yy[:,1:2,...].min())/(yy[:,1:2,...].max()-yy[:,1:2,...].min()),0,1), ep, dataformats='NCHW')
        if print_target == False:
            writer.add_images('target_real', (yy[:,0:1,...]-yy[:,0:1,...].max())/(yy[:,0:1,...].max()-yy[:,0:1,...].min()), ep, dataformats='NCHW')
            writer.add_images('target_imag', (yy[:,1:2,...]-yy[:,1:2,...].max())/(yy[:,1:2,...].max()-yy[:,1:2,...].min()), ep, dataformats='NCHW')
            # print_target = True   ################################ WARNING, remember to delete comment ##############################

        writer.add_scalar('RMSE', valid_mse, ep)
        if valid_mse < min_valid_mse and ep > 50:
            torch.save(model, os.path.join(path_model, "ep_" + str(ep) + ".pth"))
            min_valid_mse = valid_mse


        if (ep+1) % 50 == 0:
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'np_rand_state': np.random.get_state(),
                        'scheduler': scheduler.state_dict(),
                        'torch_rand_state': torch.get_rng_state(),
                        }, os.path.join(path_model,"checkpoint.pth"))

        scheduler.step()

        t2 = default_timer()
        print(ep, t2 - t1, train_mse, valid_mse)



    if os.path.isfile(os.path.join(path_model, "checkpoint.pth")):
        os.remove(os.path.join(path_model, "checkpoint.pth"))
    torch.save(model, os.path.join(path_model, "ep_" + str(epochs) + "_final.pth"))



if __name__ == '__main__':
    main()
# %%