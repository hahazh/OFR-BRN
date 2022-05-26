
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from OFR_BRN import Flow_STSR
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.nn import functional as F

def demo_test():
   
   
    os.environ["CUDA_VISIBLE_DEVICES"]='5'
    model = Flow_STSR()
    model = model.cuda()
    model.eval()
    
  
    model_dict =  torch.load(args.weight)
    model.load_state_dict(model_dict , strict=True)


    base = args.datapath
    out_p = args.outputpath
    base_ls = sorted(os.listdir(base))
    for each in base_ls:
        sub_p = os.path.join(base,each)
        sub_ls = sorted(os.listdir(sub_p))
        sub_img_ls = []
        for i in range(len(sub_ls)):
            # print(sub)
            img_p = os.path.join(base,each,sub_ls[i])
            img = cv2.imread(img_p).astype('float32')/255.0
            img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
            sub_img_ls.append(img)
        # sub_img_ls+=sub_img_ls[::-1]
        img_ten = torch.stack(sub_img_ls,dim=1).cuda()

        n, t,c, h, w = img_ten.shape
        img_ten = img_ten.view(n*t,c,h,w)
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (pw-w-(pw - w)//2, (pw - w)//2, ph-h-(ph - h)//2, (ph - h)//2)
        img_ten = F.pad(img_ten, padding,mode='reflect')
        img_ten = img_ten.view(n,t,c,img_ten.shape[2],img_ten.shape[3])
        print(img_ten.shape)
        with torch.no_grad():
            out = model(img_ten).squeeze(0)
            out = out[:,:,padding[2]*4:(padding[2]+h)*4,4*padding[0]:4*(padding[0]+w)]
            print(out.shape)
        for ix in  range(out.shape[0]):
            this_img = out[ix,:,:,:].permute(1,2,0).detach().cpu().clamp(0,1).numpy()*255.0
            this_out = os.path.join(out_p,each)
            if not os.path.exists(this_out):
                os.makedirs(this_out)
            # if ix%2==0:
            cv2.imwrite(this_out+'/'+str(ix).zfill(3)+'.png',this_img)   

if __name__ == "__main__":
    parser = ArgumentParser(description="validation script for reds",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', default='./demo/input', type=str, help='dataset path')
    parser.add_argument('--outputpath', default='./demo/output', type=str, help='outputpath of test sequence')
    parser.add_argument('--weight', default='../pretrained_weight/ofr-brn.pth', type=str, help='weight of model')

    args = parser.parse_args()
    demo_test()
 