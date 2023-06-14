############################
#
#  Codes for "Self-supervised learning of hologram reconstruction using physics consistency"
#  Authors: Luzhe Huang, Hanlong Chen, Tairan Liu and Aydogan Ozcan
#  Contact: ozcan@ucla.edu
#
#  my_tools.py: essential functions & datasets for GedankenNet
#
############################

from os import error
from typing import List
import random
import torch
import numpy as np
import scipy
import scipy.io
import scipy.signal
import skimage.measure
from torch import tensor
import torch.nn as nn
import PIL
# import h5py

import operator
from functools import reduce
from functools import partial

from multiprocessing.pool import ThreadPool
import time

# from randimage import get_random_image

z_positions_list = []


def min_max_norm(img, vmin=None, vmax=None):
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()
    img = np.clip(img, vmin, vmax)
    return (img - vmin) / (vmax - vmin)

# normalize a complex field by dividing its complex mean
def comp_field_norm(comp_field):
    # comp_field [N, C=2 (real&imag), H, W] or [C, H, W]
    if isinstance(comp_field, np.ndarray):
        if comp_field.ndim == 3:
            comp_field = comp_field[0,...] + 1j*comp_field[1,...]
            comp_field /= (np.mean(np.abs(comp_field), axis=(-2,-1), keepdims=True) * np.exp(1j*np.mean(np.angle(comp_field), axis=(-2,-1), keepdims=True)))
            comp_field /= np.mean(comp_field, axis=(-2,-1), keepdims=True)
            return np.stack((np.real(comp_field), np.imag(comp_field)), axis=0)
        elif comp_field.ndim == 4:
            comp_field = comp_field[:,0,...] + 1j*comp_field[:,1,...]
            comp_field /= (np.mean(np.abs(comp_field), axis=(-2,-1), keepdims=True) * np.exp(1j*np.mean(np.angle(comp_field), axis=(-2,-1), keepdims=True)))
            comp_field /= np.mean(comp_field, axis=(-2,-1), keepdims=True)
            return np.stack((np.real(comp_field), np.imag(comp_field)), axis=1)
    elif isinstance(comp_field, torch.Tensor):
        comp_field = comp_field[:,0,...] + 1j*comp_field[:,1,...]
        comp_field /= (torch.mean(torch.abs(comp_field), dim=(-2,-1), keepdim=True) * torch.exp(1j*torch.mean(torch.angle(comp_field), dim=(-2,-1), keepdim=True)))
        comp_field /= torch.mean(comp_field, dim=(-2,-1), keepdim=True)
        return torch.stack([torch.real(comp_field), torch.imag(comp_field)], dim=1)

# enhanced correlation coefficient
def ECC(im, im_ref, centralize=True):
    if centralize:
        im -= im.mean()
        im_ref -= im_ref.mean()
    corr = np.real(np.vdot(im, im_ref)) / np.sqrt((np.abs(im)**2).sum() * (np.abs(im_ref)**2).sum())
    return corr


# Gedanken dataset to simulate holograms using artificial images
class GedankenDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, z_list, trans, params) -> None:
        self.file_paths = file_paths
        self.trans = trans
        self.z_list = z_list
        self.params = params

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        
        amp_img = np.array(PIL.Image.open(self.file_paths[index])).astype('float32')/255
        amp = self.trans(amp_img[:,:,0]).numpy().squeeze()
        ang_img = np.array(PIL.Image.open(self.file_paths[index//2])).astype('float32')/255
        ang = self.trans(ang_img[:,:,0]).numpy().squeeze()
        s = amp.shape[-1]
        comp_field = (amp+0.05) * np.exp(1j*1.0*np.pi*ang)  # 0.05 is DC
        # 2x downsampling & upsampling
        comp_field = skimage.measure.block_reduce(comp_field, block_size=2, func=np.mean)
        comp_field = scipy.ndimage.zoom(comp_field, 2, order=1)
        # Gaussian smooth
        comp_field = scipy.ndimage.gaussian_filter(comp_field, sigma=1.0, mode='constant', cval=0)
        # add white noise
        comp_field += np.random.normal(loc=0, scale=0.005*np.sqrt(2), size=(s,s,2)).view(np.complex128).reshape(s,s).astype(np.complex64)  # 40 dB

        # comp_field /= comp_field.mean()  # DO NOT NORMALIZE if phase in [0, 2pi]
        re, im = np.real(comp_field), np.imag(comp_field)
        wave = AxialPlane(comp_field.squeeze(), 0, self.params)
        prop_field = np.stack([np.stack(wave(z), axis=-1) for z in self.z_list], axis=0)  # [T, H, W, C=2]
        inp = np.sqrt(np.sum(np.square(prop_field), axis=-1)).squeeze().astype('float32')  # [T, H, W]
        # inp += np.random.normal(0, 0.1, inp.shape)
        if inp.ndim == 2:
            inp = inp[np.newaxis,...]
        tag = np.stack((re, im), axis=0)  # [C=2, H, W]
        # tag = gt_field.transpose([2,0,1])  # [C=2, H, W]
        return torch.Tensor(inp), torch.Tensor(tag)


# COCO dataset to simulate holograms using COCO natural images
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, z_list, trans, params, mode='P') -> None:
        self.imgs = file_paths
        self.trans = trans
        self.inp = []
        self.tag = []
        self.z_list = z_list
        self.mode = mode
        self.params = params

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        amp_img = np.array(PIL.Image.open(self.imgs[index])).astype('float32')/255
        if amp_img.ndim == 3:
            amp_img = np.matmul(amp_img, [0.2989, 0.5870, 0.1140])  # [H, W]
        h, w = amp_img.shape
        if h <= 512 or w <= 512:
            amp_img = scipy.ndimage.zoom(amp_img, (max(1, 512/h), max(1, 512/w)))
        # amp_img = amp_img / amp_img.mean()
        amp = self.trans(amp_img)
        ang_img = np.array(PIL.Image.open(self.imgs[index//2])).astype('float32')/255
        if ang_img.ndim == 3:
            ang_img = np.matmul(ang_img, [0.2989, 0.5870, 0.1140])  # [H, W]
        h, w = ang_img.shape
        if h <= 512 or w <= 512:
            ang_img = scipy.ndimage.zoom(ang_img, (max(1, 512/h), max(1, 512/w)))
        ang = self.trans(ang_img)
        # ang = 2*np.pi*(self.trans(ang_img) - 1/2)
        if self.mode == 'P':
            comp_field = (amp+0.05) * np.exp(1j*np.pi*ang)
            comp_field /= comp_field.mean()
            re, im = np.real(comp_field), np.imag(comp_field)
        elif self.mode == 'R':
            comp_field = amp + 1j*ang
            comp_field /= comp_field.mean()
            re, im = np.real(comp_field), np.imag(comp_field)
        wave = AxialPlane(comp_field.squeeze(), 0, self.params)
        prop_field = np.stack([np.stack(wave(z), axis=-1) for z in self.z_list], axis=0)  # [T, H, W, C=2]
        inp = np.sqrt(np.sum(np.square(prop_field), axis=-1)).squeeze().astype('float32')  # [T, H, W]
        # inp += np.random.normal(0, 0.1, inp.shape)
        if inp.ndim == 2:
            inp = inp[np.newaxis,...]
        tag = np.concatenate((re, im), axis=0)  # [C=2, H, W]
        # tag = gt_field.transpose([2,0,1])  # [C=2, H, W]
        return torch.Tensor(inp), torch.Tensor(tag)


# experimental hologram dataset that loads holograms from .mat files
class HoloDataset(torch.utils.data.Dataset):  # for experimental dataset
    def __init__(self, file_paths, z_idx, trans) -> None:
        self.gt_images = file_paths
        self.trans = trans
        self.inp = []
        self.tag = []
        self.z_idx = z_idx

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, index):
        tmp = scipy.io.loadmat(self.gt_images[index])
        # tmp = h5py.File(self.gt_images[index],'r')
        gt_field = tmp['targetData'].astype('float32')  # [H, W, C=2(real&imag)]
        inp = tmp['inputData'][:,:,self.z_idx]  # [H, W, D]
        img = np.concatenate((inp, gt_field), axis=-1)
        img = self.trans(img)
        inp = img[:-2,...]  # [T, H, W]
        tag = img[-2:,...]  # [C=2, H, W]
        # inp += np.random.normal(0, 0.1, inp.shape)
        # inp = inp.transpose([1,2,0])  # [H, W, T]
        # tag = gt_field.transpose([2,0,1])  # [C=2, H, W]
        return inp, tag


class AxialPlane():
    def __init__(self, init_guess, z0, params) -> None:

        # parameters
        self.z0 = z0
        self.wl = params['wavelength'] / params['ref_ind']  # effective wavelength, [um]
        self.k = 1 / self.wl
        self.n_pixel = params['patch_size']
        self.pixel_size = params['pixel_size']  # [um]
        self.max_freq = 1/self.pixel_size  # [um-1]

        # initialize complex field
        h, w = init_guess.shape[-2], init_guess.shape[-1]
        if h != params['patch_size'] or w != params['patch_size']:
            init_guess = np.pad(init_guess, (((self.n_pixel-h)//2, self.n_pixel-h-(self.n_pixel-h)//2), ((self.n_pixel-w)//2, self.n_pixel-w-(self.n_pixel-w)//2)))
        self.comp_field = init_guess
        self.h, self.w = h, w

    def __call__(self, z):

        # free space propagation using angular spectrum
        ang_spectrum = np.fft.fftshift(np.fft.fft2(self.comp_field))

        # create grid                                                    
        uu, vv = np.meshgrid(self.max_freq*np.arange(-np.ceil((self.n_pixel-1)/2), np.floor((self.n_pixel+1)/2))/self.n_pixel,
                    self.max_freq*np.arange(-np.ceil((self.n_pixel-1)/2), np.floor((self.n_pixel+1)/2))/self.n_pixel)
        mask = ((self.k**2 - uu**2 - vv**2) >= 0).astype('float32')
        ww = np.sqrt((self.k**2-uu**2-vv**2)*mask).astype('float32')
        # transfer function
        dz =z - self.z0
        h = np.exp(1j * 2 * np.pi * ww * dz) * mask
        ang_spectrum *= h

        # IFFT
        prop_field = np.fft.ifft2(np.fft.ifftshift(ang_spectrum))
        prop_field = prop_field[(self.n_pixel-self.h)//2:(self.n_pixel-self.h)//2+self.h, (self.n_pixel-self.w)//2:(self.n_pixel-self.w)//2+self.w]

        return prop_field.real, prop_field.imag

def P2C(radii, angles):
    # exponential (amp & phase) to complex
    return radii * torch.exp(1j*angles)

def C2P(x):
    # complex to exponential (amp & phase)
    return torch.abs(x), torch.angle(x)

def R2P(x):
    # real & imag to exponential (amp & phase)
    if isinstance(x, torch.Tensor):
        return torch.sqrt(torch.sum(torch.square(x), axis=1)), torch.atan2(x[:,1,...], x[:,0,...])
    elif isinstance(x, np.ndarray):
        if x.ndim == 3:
            return np.sqrt(np.sum(np.square(x), axis=0)), np.arctan2(x[1,...], x[0,...])
        elif x.ndim == 4:
            return np.sqrt(np.sum(np.square(x), axis=1)), np.arctan2(x[:,1,...], x[:,0,...])