from ast import arg
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from OFR_BRN import Flow_STSR
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mydata.load_data import get_REDS_loader


def get_model_total_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return (1.0*params/(1000*1000))

def main():

    
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    model = Flow_STSR()
    model = model.cuda()
    model.eval()

    print("para ",get_model_total_params(model))

    scale = 4

    batch_size = 1
    seq_length = 13
    test_loader = get_REDS_loader(args.datapath,  scale,seq_length,batch_size,num_workers=0, shuffle=False,mode='mirror' )
  
    model_dict =  torch.load(args.weight)
    model.load_state_dict(model_dict , strict=True)
    
    
    with torch.no_grad():
        for i, (gt_image,images ,tag,crop_shape) in enumerate(tqdm(test_loader)):

          
            tag_seq = tag[0][0].split('/')[-2]
            tag_ls = [each[0].split('/')[-2]+'/'+each[0].split('/')[-1] for each in tag]

            images = images.cuda()
           
            out = model(images)
            if  not os.path.exists(args.outputpath+'/'+tag_seq):
                os.makedirs(args.outputpath+'/'+tag_seq)

            out_cp = out.squeeze(0)

            this_ls = []
            x1,x2,y1,y2 = 0,0,0,0
            for j in range(seq_length):
                img = out_cp[j].permute(1,2,0).detach().cpu().clamp(0,1).numpy()*255.0
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x1,x2,y1,y2 = crop_shape
                img = img[x1:x2,y1:y2,:]
                this_ls.append(tag_ls[j])
             
                cv2.imwrite(args.outputpath+'/'+tag_ls[j],img)
  
         
           

if __name__ == "__main__":
    parser = ArgumentParser(description="validation script for reds",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', default='xxx', type=str, help='dataset path')
    parser.add_argument('--outputpath', default='../output/reds', type=str, help='outputpath of test sequence')
    parser.add_argument('--weight', default='../pretrained_weight/ofr-brn.pth', type=str, help='weight of model')

    args = parser.parse_args()
    main()
 