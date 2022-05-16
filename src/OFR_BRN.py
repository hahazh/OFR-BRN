

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from modules.general_module import ResidualBlocksWithInputConv, PixelShufflePack

from modules.general_module import flow_warp

from modules.IFnet_m import IFNet
from modules.correlation_module import Correlation_cal

class Flow_STSR(nn.Module):
   

    def __init__(self,
                 mid_channels=64,
                 num_blocks=30,
                 padding=2,
                ):

        super().__init__()

        self.mid_channels = mid_channels
        self.padding = padding

        self.IFNet = IFNet()

        self.corr_adaptive = Correlation_cal()
       
        self.conv_first = nn.Conv2d(3, 3 * 3, 3, 1, 1)
      
        self.backward_branch = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)

        self.forward_branch = ResidualBlocksWithInputConv(
            2 * mid_channels + 3, mid_channels, num_blocks)

        # upsample
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
  
        self.mix_blend =  nn.Conv2d(2*64+3, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def compute_biflow(self, lrs):

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        lrs_catted = torch.cat((lrs_1,lrs_2),dim=1)

        flows_out = self.IFNet(lrs_catted).view(n, 2*(t - 1), 2, h, w)
        backward_half_ix = [2*ix+1 for ix in range(t-1)]
        forward_half_ix = [2*ix for ix in range(t-1)]
        flow_out_backward_half = flows_out[:,backward_half_ix,:,:,:]
        flow_out_forward_half = flows_out[:,forward_half_ix,:,:,:] 
        #reverse first flow
        flows_backward = -flow_out_forward_half + flow_out_backward_half
        for i in range(t-1):
            flows_backward[:,i,:,:,:] = flow_warp(flows_backward[:,i,:,:,:],flow_out_backward_half[:,i,:,:,:].permute(0, 2, 3, 1))
    
        #reverse second flow
        flows_forward = -flow_out_backward_half + flow_out_forward_half
        for i in range(t-1):
            flows_forward[:,i,:,:,:] = flow_warp(flows_forward[:,i,:,:,:],flow_out_forward_half[:,i,:,:,:].permute(0, 2, 3, 1))

        return flows_forward, flows_backward,flow_out_forward_half,flow_out_backward_half

    def forward(self, lrs):
 
        # ----------get inserted feature-----------

        n, t, c, h_input, w_input = lrs.size()
        h, w = lrs.size(3), lrs.size(4)
        flows_forward, flows_backward,flow_out_forward_half,flow_out_backward_half = self.compute_biflow(lrs)

        outputs = []
        outputs_lr = []
        mix_out = []

        HSF = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            # ----------------for lrs ---------------
            lr_curr = lrs[:, i, :, :, :]

            if i < t - 1:  
                flow = flows_backward[:, i, :, :, :]
                HSF = flow_warp(HSF, flow.permute(0, 2, 3, 1))
           
            if i< t-1:
                lr_fea =  self.conv_first(lr_curr)
                HSF = self.corr_adaptive(HSF,lr_fea)
            HSF = torch.cat([lr_curr, HSF], dim=1)
            HSF = self.backward_branch(HSF)
            outputs.append(HSF)
            # ----------------for inserted lrs ---------------
  
            if i < t - 1:
                # get warped_lr
                flow = flows_backward[:, i, :, :, :]
                flow_half = flow_out_backward_half[:,i,:,:,:]
             
                warped_lr_img,img_mask = flow_warp(lrs[:, i + 1, :, :, :], flow_half.permute(0, 2, 3, 1),ret_mask=True)
                insert_HSF,feat_mask = flow_warp(outputs[-2], flow_half.permute(0, 2, 3, 1),ret_mask=True)
                insert_HSF = torch.cat([warped_lr_img, insert_HSF], dim=1)
                insert_HSF = self.backward_branch(insert_HSF)
                outputs_lr.append([warped_lr_img,insert_HSF,img_mask,feat_mask])
        for ix in range(2 * t - 1):
            if ix % 2 == 0:
                mix_out.append(outputs[ix // 2])
            else:
                mix_out.append(outputs_lr[ix // 2])
        outputs = mix_out[::-1]

        HSF = torch.zeros_like(HSF)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                previous_insert_HSF = HSF.clone()

                HSF = flow_warp(HSF, flow.permute(0, 2, 3, 1))
            if i>0 :
                lr_fea =  self.conv_first(lr_curr)
                HSF = self.corr_adaptive(HSF, lr_fea)
            HSF = torch.cat([lr_curr, outputs[2 * i], HSF], dim=1)
            HSF = self.forward_branch(HSF)

            out = self.lrelu(self.upsample1(HSF))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
         
            outputs[i * 2] = out

            # ----------------for inserted lrs ---------------
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                flow_half = flow_out_forward_half[:, i - 1, :, :, :]
             
                warped_lr_img,mask_img2 = flow_warp(lrs[:, i - 1, :, :, :],flow_half.permute(0, 2, 3, 1),ret_mask=True)
              
                insert_HSF,mask_feat2 = flow_warp(previous_insert_HSF, flow_half.permute(0, 2, 3, 1),ret_mask=True)
              
                output_img = outputs[2 * i - 1][0]
                outputs_feat = outputs[2 * i - 1][1]
                outputs_img_mask1 = outputs[2 * i - 1][2]
                outputs_feat_mask1 = outputs[2 * i - 1][3]

                frame_mix_ = (output_img + warped_lr_img) / 2
                frame_mix = frame_mix_ + 0.5 * (warped_lr_img * (1-outputs_img_mask1) + output_img * (1-mask_img2))
                feat_mix_ = (outputs_feat+insert_HSF)/2
                feat_mix = feat_mix_ + 0.5 * (insert_HSF * (1-outputs_feat_mask1) + outputs_feat * (1-mask_feat2))

                insert_HSF = torch.cat([warped_lr_img,outputs_feat , insert_HSF], dim=1)
                insert_HSF = self.forward_branch(insert_HSF)
                #mix 
                final_feat = self.mix_blend(torch.cat((frame_mix,feat_mix,insert_HSF),dim=1))

                out_insert = self.lrelu(self.upsample1(final_feat))
                out_insert = self.lrelu(self.upsample2(out_insert))
                out_insert = self.lrelu(self.conv_hr(out_insert))
                out_insert = self.conv_last(out_insert)
                outputs[2 * i - 1] = out_insert

        return torch.stack(outputs, dim=1)[:, :, :, :4 * h_input, :4 * w_input]

 


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    (n, t, c, h, w) = 1, 3, 3, 64, 64
    fstsr = Flow_STSR()
  
    ivsr = fstsr.cuda()
    in_data = torch.zeros(n, t, c, h, w)
    in_data = in_data.cuda()
    out_data = ivsr(in_data)
    print(out_data.shape)
