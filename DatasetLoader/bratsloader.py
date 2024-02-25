import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import random



class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    if "npy" in f:
                        continue
                    seqtype = f.split('_')[3][:-4]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
        self.cache = dict()

    def __getitem__(self, x):
        out = []
        filedict = self.database[99]
        a = "_" 
        npy = a.join(filedict["flair"].split('_')[:-1]) + ".npy"
        if os.path.exists(npy):
            out = torch.tensor(np.load(npy))
        else:
            for seqtype in self.seqtypes:
            # if filedict[seqtype] in self.cache.keys():
            #     out.append(self.cache[filedict[seqtype]])
            # else:
            #     nib_img = nibabel.load(filedict[seqtype])
            #     path=filedict[seqtype]
            #     fd = torch.tensor(nib_img.get_fdata())
            #     out.append(fd)
            #     self.cache[filedict[seqtype]] = fd
                nib_img = nibabel.load(filedict[seqtype])
                path=filedict[seqtype]
                out.append(torch.tensor(nib_img.get_fdata()))
            out = torch.stack(out)         
            np.save(npy,out)
        out = out.cuda()
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:
            
            image = out[:-2, ...].permute(0,3,1,2)
            label = out[-1, ...][None, ...].permute(0,3,1,2)
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, -1).float()  #merge all tumor classes into one
            image = image / (torch.max(image))
            #rand = torch.randint(0, 154, (1 , 8))
            rand = torch.tensor([[30,55,66,75,79,82,128,142]])
            image = image.permute(1,0,2,3)[rand[0]]
            label = label.permute(1,0,2,3)[rand[0]]
            if self.transform:
            #     state = torch.get_rng_state()
                image = self.transform(image)
            #     torch.set_rng_state(state)
                label = self.transform(label)
            return (image.cpu(), label.cpu(), 99, rand[0])

    def __len__(self):
        return len(self.database)