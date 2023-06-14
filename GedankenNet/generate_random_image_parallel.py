############################
#
#  Codes for "Self-supervised learning of hologram reconstruction using physics consistency"
#  Authors: Luzhe Huang, Hanlong Chen, Tairan Liu and Aydogan Ozcan
#  Contact: ozcan@ucla.edu
#
#  generate_random_image_parallel.py: generate 100K artificial images for GedankenNet training
#
############################

import multiprocessing
import randimage
import matplotlib.pyplot as plt
import numpy as np
import os

wt_dir = ''  # SPECIFY THE WRITING DIRECTORY HERE FOR THE 100K IMAGES

def gen_random_image(i):
    tmp = randimage.get_random_image((512,512))
    tmp = np.matmul(tmp,  [0.2989, 0.5870, 0.1140])  # convert to RGB
    plt.imsave(os.path.join(wt_dir, '%04d.png'%i), tmp, cmap='gray')
    return i

def main():
    pool = multiprocessing.Pool(12)  # NO. OF POOLS NEED TO BE ADJUSTED BASED ON YOUR HARDWARE
    ii = pool.map(gen_random_image, range(100000))

if __name__ == '__main__':
    main()