# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from model.network import MainModel
from DatasetLoader.isicloader import ISICDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from medpy import metric
import os

os.environ["https_proxy"]="http://127.0.0.1:7890"
os.environ["http_proxy"]="http://127.0.0.1:7890"

def worker_init_fn(worker_id):
    random.seed(worker_id)

def main():
    # Setup PyTorch:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_sampling_steps = 300

    # Load model:
    image_size = 256
    assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
    model = MainModel().to(device)
    diffusion = create_diffusion(str(num_sampling_steps), diffusion_steps=300)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # DATASET
    tran_list = [transforms.Resize((256,256)), transforms.ToTensor(),]
    transform_train = transforms.Compose(tran_list)
    trainSet = ISICDataset(None, "F:\DiT\dataset", transform_train)
    testSet = ISICDataset(None, "F:\DiT\dataset", transform_train, mode = "Test")

    tran_list = [transforms.Resize((256,256)),]
    transform_train = transforms.Compose(tran_list)
    trainloader = DataLoader(trainSet, batch_size=8, num_workers=0,shuffle=True,  pin_memory=True)
    testloader = DataLoader(testSet, batch_size=8, num_workers=0, shuffle=True,  pin_memory=True)

    iterator = tqdm(range(500), ncols=70)
    for epoch_num in iterator:       
        # train Part
        file = open("./result.tsv", "a")
        model.train()
        allloss = 0
        count = 0
        for datanum, data in enumerate(trainloader):
            image_batch_size = data[0].shape[0]
            image = data[0].float().cuda()
            mask = data[1].float().cuda()
            z = vae.encode(x = mask).latent_dist.sample()
            y = image
            # y = vae.encode(x = image).latent_dist.sample()
            if z.shape[0] % 2 == 0:
                t = torch.tensor(np.random.randint(num_sampling_steps, size=z.shape[0] // 2)).long()
                t = torch.concat([t, num_sampling_steps - t - 1]).cuda()
            else:
                t = torch.tensor(np.random.randint(num_sampling_steps, size=z.shape[0])).long().cuda()

            model_kwargs = dict(y=y)
            loss = diffusion.training_losses(
                model, z, t, model_kwargs=model_kwargs, noise = None
            )["loss"].mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            allloss += loss.item()
            count += 1
            print("epoch:%d  loss: %f t: %d"%(datanum, loss.item(), t[0]))  
        print("epoch:%d  loss: %f"%(epoch_num, allloss / count))   
        epoch_loss = allloss / count

        # train validation
        torch.set_grad_enabled(False)
        dice=0
        count = 0
        for datanum, data in enumerate(trainloader): 
            image_batch_size = data[0].shape[0] 
            image = data[0].float().cuda()
            mask = data[1].float().cuda()
            z = torch.randn(image_batch_size, 4, 32, 32, device=device)
            y = image
            #y = vae.encode(x = image).latent_dist.sample()
            model_kwargs = dict(y=y)
            samples = diffusion.p_sample_loop(
                    model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples = vae.decode(samples / 0.18215).sample
            
            output = torch.where(samples.detach() < 0.2, 0, 1).cpu().float()
            label = torch.where(mask.detach() <0.2, 0, 1).cpu().float()
            dice += metric.binary.dc(np.array(output, dtype=bool), np.array(label, dtype=bool))
        print("dice:%d ", dice/count)       
        epoch_dice_train = dice / count

        # test validation
        dice=0
        count = 0
        for datanum, data in enumerate(testloader): 
            image_batch_size = data[0].shape[0] 
            image = data[0].float().cuda()
            mask = data[1].float().cuda()
            z = torch.randn(image_batch_size, 4, 32, 32, device=device)
            y = image
            #y = vae.encode(x = image).latent_dist.sample()
            model_kwargs = dict(y=y)
            samples = diffusion.p_sample_loop(
                    model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples = vae.decode(samples / 0.18215).sample
            
            output = torch.where(samples.detach() < 0.2, 0, 1).cpu().float()
            label = torch.where(mask.detach() < 0.2, 0, 1).cpu().float()
            dice += metric.binary.dc(np.array(output, dtype=bool), np.array(label, dtype=bool))
        print("dice:%d ", dice/count)       
        torch.set_grad_enabled(True)
        epoch_dice_test = dice / count
        
        file.write(f"{epoch_num}\t{epoch_loss}\t{epoch_dice_train}\t{epoch_dice_test}\n\r")
        file.close()
        if epoch_num % 200 == 199:
            states = {"model" : model.state_dict(), "optimizer" : optimizer.state_dict()}
            torch.save(states, "weights/SwinDiff_kit_epoch" + str(epoch_num) + ".pth")

    states = {"model" : model.state_dict(), "optimizer" : optimizer.state_dict()}
    torch.save(states, "weights/SwinDiff_Brats.pth")

if __name__ == '__main__':
    main()