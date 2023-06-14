# Self-supervised learning of hologram reconstruction using physics consistency

Luzhe Huang, Hanlong Chen, Tairan Liu, Aydogan Ozcan

## Environment requirements
The codes was tested on Windows 11 and Linux, with Python and PyTorch. Packages required to reproduce the results can be found in `requirements.txt`. The following software / hardware is tested and recommended:
- Python >= 3.9
- CUDA 11.2
- cuDNN >= 8.0
- Intel i9-12900F
- Nvidia RTX 3090
- RAM >= 64 GB

## File structure
This repository contains codes for GedankenNet and GedankenNet-Phase, and demo models and data for each network.
```
GedankenNet
|   README.md
|   requirements.txt
|
|---GedankenNet
|   |   generate_random_image_parallel.py
|   |   train_Gedanken.py
|   |   test.py
|   |   ...
|
|---GedankenNet_Phase
|   |   train_GedankenP.py
|   |   testP.py
|   |   ...
|
|---Models
|
|---demo_data
|   |   stained_tissue
|   |   unstained_tissue
```
`/GedankenNet` and `/GedankenNet_Phase` contain the codes to train and test the two models. 

`/Models` include two demo models for GedankenNet and GedankenNet-Phase respectively. 

`/demo_data/stained_tissue` include two demo FOVs of lung and Pap smears (complex object), corresponding to the results of `/GedankenNet`.

`/demo_data/unstained_tissue` include two demo FOVs of kidney tissue (phase-only object), corresponding to the results of `/GedankenNet_Phase`.

## Test
To test the two demo models (download via [Google Drive](https://drive.google.com/drive/folders/1q3DDUWeEky48ebyYWChalvZxzqrhtz-o?usp=sharing)) and reproduce some results shown in the paper, following these steps:
- Download checkpoints in `/Models` and mat files in `/demo_data`
- Change `TEST_PATH` in `test.py` or `testP.py` to the path of corresponding mat files 
- Run `test.py` or `testP.py`. The outputs will be saved in `outputs` folders

## Train
To train GedankenNet and GedankenNet-Phase models from scratch, follow these steps:
- Run `/GedankenNet/generate_random_image_parallel.py` to generate 100K artificial images, and then split these images into train, valid and test sets
- Change `TRAIN_PATH` and `VALID_PATH` to the locations of the generated artificial image datasets in `train_Gedanken.py` or `train_GedankenP.py`
- Run `train_Gedanken.py` or `train_GedankenP.py`, checkpoints will be saved in `Models` folders

Alternatively, we have uploaded one artificial image dataset with 99.9K training and 100 testing / validation images. Download from [Google Drive](https://drive.google.com/file/d/1kfzGJYmC8-tbUXu_BRz1CFB2Rmr17gQL/view?usp=sharing).