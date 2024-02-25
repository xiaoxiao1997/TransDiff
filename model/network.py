from model.multiTrans import DiT_B_2
import torch.nn as nn
from model.swintransformer import PatchEmbed, BasicLayer, PatchMerging


class MainModel(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=4,
    ):
        super().__init__()
        self.in_channels = in_channels      
        self.dit = DiT_B_2(input_size = 32, in_channels = in_channels, cond_channels = 256)
        self.cond_start = PatchEmbed(img_size=256, patch_size=1, in_chans=3, embed_dim=32,norm_layer=nn.LayerNorm)
        self.cond_drop = nn.Dropout(p=0.01)
        self.cond1 = BasicLayer(dim=32,
                               input_resolution=(256,256),
                               depth=2,
                               num_heads=4,
                               window_size=8,
                               downsample=PatchMerging)
        self.cond2 = BasicLayer(dim=64,
                               input_resolution=(128, 128),
                               depth=2,
                               num_heads=4,
                               window_size=8,
                               downsample=PatchMerging)
        self.cond3 = BasicLayer(dim=128,
                               input_resolution=(64, 64),
                               depth=2,
                               num_heads=4,
                               window_size=8,
                               downsample=PatchMerging)

    def forward(self, x, t, y):
        y = self.cond_start(y)
        y = self.cond_drop(y)
        y = self.cond1(y)
        y = self.cond2(y)
        y = self.cond3(y)
        y = y.view(y.shape[0],32,32,-1)
        y = y.permute(0,3,1,2) #B,C,H,W
        x = self.dit(x,t,y)
        return x