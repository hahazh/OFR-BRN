
from typing import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F



class Correlation_cal(nn.Module):
    
    def __init__(self):
        super(Correlation_cal, self).__init__()
        self.res_connect = True
        self.patch_size = 3
        self.new_ch = self.patch_size ** 2
        self.conv_first = nn.Conv2d(9, 64, 1, 1, 0)
        if self.res_connect:
            self.fuse = nn.Conv2d(64 * 2, 64, 1, 1, 0)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, hs, ref_LR):

        B, C, H, W = hs.size()
        ref_LR = self.conv_first(ref_LR)
        ref_LR = self.lrelu(ref_LR)
        ref_LR = F.pad(ref_LR, [1, 1, 1, 1], mode='reflect')
        hs_un = F.pad(hs, [1, 1, 1, 1], mode='reflect')
        ref_LR = F.unfold(input=ref_LR, kernel_size=(3, 3), dilation=1, padding=0, stride=1)
        hs_un = F.unfold(input=hs_un, kernel_size=(3, 3), dilation=1, padding=0, stride=1)
        ref_LR = ref_LR.view(B, -1, self.new_ch , H, W)
        hs_un = hs_un.view(B, C, -1, H, W)
        hs_un = hs_un.permute(0, 3, 4, 1, 2).unsqueeze(4)
        ref_LR = ref_LR.permute(0, 3, 4, 1, 2).unsqueeze(5)
        cross_product = hs_un @ ref_LR
        cross_product = cross_product.squeeze(dim=4).squeeze(dim=4).permute(0,3,1,2)
        att_matrix = F.sigmoid(cross_product)
        if self.res_connect:
            # Hadamard product
            corr_res = att_matrix * hs
            corr_res = torch.cat((corr_res, hs), dim=1)
            corr_res = self.fuse(corr_res)

        else:
            corr_res = att_matrix * hs
        return corr_res

