import os
import random
import json
import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image.permute(1,2,0), k)
    label = np.rot90(label.permute(1,2,0), k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return torch.tensor(image).permute(2,0,1), torch.tensor(label).permute(2,0,1)


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


class KIT_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        content = ""
        with open(os.path.join(base_dir,'kits.json'),'r',encoding='utf-8') as f:
            content = f.read()
        self.sample_list = json.loads(content) #open(os.path.join(list_dir, self.split+'.txt')).readline()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]["case_id"]
        slice_name = "case_00192"
        data_path = os.path.join(self.data_dir, slice_name, 'imaging.nii.gz')
        label_path = os.path.join(self.data_dir, slice_name, 'segmentation.nii.gz')
        data = torch.tensor(nib.load(data_path).get_fdata())
        label = torch.tensor(nib.load(label_path).get_fdata())
        rand = torch.randint(0, data.shape[0] - 1, (1 , 8))
        rand = torch.tensor([[65, 66, 69, 74, 75, 77, 82, 85]])
        data = data[rand[0]].float()
        label = label[rand[0]].float()
        data= (data + 1024) / 2048
        label=torch.where(label > 0, 1, 0).float()
        # data, label = random_rot_flip(data, label)
        if self.transform:
            data = self.transform(data)
            label = self.transform(label)       
        sample = [data, label, slice_name, rand]
        # sample['case_name'] = slice_name
        return sample
