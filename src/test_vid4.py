from ast import arg
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from OFR_BRN import Flow_STSR
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mydata.load_data import get_vid4_loader


def get_model_total_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return (1.0*params/(1000*1000))

def main():

    
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    model = Flow_STSR()
    model = model.cuda()
    model.eval()

    print("para ",get_model_total_params(model))
    seq_name_tuple = ('calendar','city','foliage','walk')
    scale = 4
    seq_length_list = (41,33,49,47)
    batch_size = 1
    model_dict =  torch.load(args.weight)
    model.load_state_dict(model_dict , strict=True)

    
    with torch.no_grad():
        for q in range(4):
            tag = seq_name_tuple[q]
            seq_length = seq_length_list[q]
            vid4_dataloader = get_vid4_loader(args.datapath,scale,tag,seq_length,batch_size,0)
            for i, (gt_image,images,tag_ix,crop_shape) in enumerate(tqdm(vid4_dataloader)):

                
                images = images.cuda()


                out = model(images)

                img_p = args.outputpath+'/'+tag
                if   not os.path.exists(img_p):
                    os.makedirs(img_p)
                out_cp = out.squeeze(0)
            
                for j in range(seq_length):
                    img = out_cp[j].permute(1,2,0).detach().cpu().clamp(0,1).numpy()*255.0
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    x1,x2,y1,y2 = crop_shape
                    
                    img = img[x1:x2,y1:y2,:]
                    cv2.imwrite(img_p+'/'+str(int(tag_ix[j])).zfill(8)+'.png',img)



if __name__ == "__main__":
    parser = ArgumentParser(description="validation script for vid4",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', default='xxx', type=str, help='dataset path')
    parser.add_argument('--outputpath', default='../output/vid4', type=str, help='outputpath of test sequence')
    parser.add_argument('--weight', default='../pretrained_weight/ofr-brn.pth', type=str, help='weight of model')

    args = parser.parse_args()
    main()
 