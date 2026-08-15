import torch
import torch.nn as nn
import torch.nn.functional as F
from .nafnet import NAFNet

class NAFNetSR(nn.Module):
    def __init__(self, img_channel=1, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], scale=2):
        super().__init__()
        self.scale = scale
        self.backbone = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        
        # Override ending
        if scale > 1:
            self.backbone.ending = nn.Sequential(
                nn.Conv2d(width, img_channel * scale * scale, kernel_size=3, padding=1),
                nn.PixelShuffle(scale)
            )

    def forward(self, inp):
        x, pad_h, pad_w = self.backbone.pad_image(inp)
        
        x = self.backbone.intro(x)
        
        encs = []
        for encoder, down in zip(self.backbone.encoders, self.backbone.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            
        x = self.backbone.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.backbone.decoders, self.backbone.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            
        x = self.backbone.ending(x)
        
        if self.scale > 1:
            base = F.interpolate(inp, scale_factor=self.scale, mode='bicubic', align_corners=False)
            x = x + base
        else:
            x = x + inp
            
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :x.shape[2]-pad_h*self.scale, :x.shape[3]-pad_w*self.scale]
            
        return x

def build_model(config):
    mc = config['model']
    return NAFNetSR(
        img_channel=mc.get('img_channel', 1),
        width=mc.get('width', 16),
        middle_blk_num=mc.get('middle_blk_num', 1),
        enc_blk_nums=mc.get('enc_blk_nums', [1,1,1,2]),
        dec_blk_nums=mc.get('dec_blk_nums', [1,1,1,1]),
        scale=mc.get('scale', 2)
    )
