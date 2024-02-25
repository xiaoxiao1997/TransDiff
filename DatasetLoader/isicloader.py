import os
import sys
import pickle
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
import random
from scipy import ndimage
from scipy.ndimage import zoom

def random_rot_flip(image, label):
    image = np.transpose(image, (1,2,0))
    label = np.transpose(label, (1,2,0))

    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    image = np.transpose(image, (2,0,1))
    label = np.transpose(label, (2,0,1))

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
    
class ISICDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]+'.jpg'
        img_path = os.path.join(self.data_path, 'ISBI2016_ISIC_Part3B_'+ self.mode +'_Data',name)
        
        mask_name = name.split('.')[0] + '_Segmentation.png'
        msk_path = os.path.join(self.data_path, 'ISBI2016_ISIC_Part3B_'+ self.mode +'_Data',mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('RGB')

        if self.mode == 'Training':
            label = 0 if self.label_list[index] == 'benign' else 1
        else:
            label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

            # if random.random() > 0.5:
            #     img, mask = random_rot_flip(img, mask)

        if self.mode == 'Training':
            return (torch.tensor(img), torch.tensor(mask))
        else:
            return (torch.tensor(img), torch.tensor(mask), name)