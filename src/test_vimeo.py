import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from OFR_BRN import Flow_STSR
from mydata.load_data import get_loader_vimeo
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def get_model_total_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return (1.0*params/(1000*1000))

""" Entry Point """
def main():
     # specify your cuda env
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    model = Flow_STSR()
    model = model.cuda()
    model.eval()
    print("para ",get_model_total_params(model))

    test_file_name = 'sep_testlist.txt'
    test_loader = get_loader_vimeo(args.datapath,test_file_name, 1, False,0)   
    model_dict =  torch.load(args.weight)
    model.load_state_dict(model_dict , strict=True)

    with torch.no_grad():
        for i, (images,tag) in enumerate(tqdm(test_loader)):
            tag = tag[0].split('/')[-2]+'/'+tag[0].split('/')[-1]
            
            images = images.cuda()
            out = model(images)
            if not os.path.exists(args.outputpath+'/'+tag):
                os.makedirs(args.outputpath+'/'+tag)
            img_p = args.outputpath+'/'+tag
            out_cp = out.squeeze(0)
            for j in range(7):
                img = out_cp[j].permute(1,2,0).detach().cpu().clamp(0,1).numpy()*255.0
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_p+'/'+''+str(j)+'.png',img)



if __name__ == "__main__":

    parser = ArgumentParser(description="validation script for vimeo",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', default='xxx', type=str, help='dataset path')
    parser.add_argument('--outputpath', default='../output/vimeo', type=str, help='outputpath of test sequence')
    parser.add_argument('--weight', default='../pretrained_weight/ofr-brn.pth', type=str, help='weight of model')

    args = parser.parse_args()
    main()