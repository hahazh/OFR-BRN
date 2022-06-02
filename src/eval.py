import os
from time import process_time
import cv2
import glob

from PIL import Image 
import cv2
import numpy as np
from os import path as osp

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr
def cal_REDS():
    GT_base = 'xxx'
    out_base = 'xxx'
    out_img_p1_dir = sorted(os.listdir(out_base))
    psnr_avg_total = 0
    ssim_avg_total = 0
    for each in out_img_p1_dir:
        print(each)
        psnr_avg_seq = 0
        ssim_avg_seq = 0
        for i in range(1,98):
            print(each+'/'+str(i).zfill(8)+'.png')
            img_p = os.path.join(out_base,each,str(i).zfill(8)+'.png')
           
            img_gt_p = os.path.join(GT_base,each,str(i).zfill(8)+'.png')
           
            img_gt = cv2.imread(img_gt_p, cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.
            img_restored = cv2.imread(
            img_p,
            cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
            psnr = calculate_psnr(
            img_gt * 255,
            img_restored * 255,
            crop_border = 0
            )
            ssim = calculate_ssim(
            img_gt * 255,
            img_restored * 255,
             crop_border = 0
            )

            psnr_avg_seq+=psnr
            ssim_avg_seq+=ssim
            psnr_avg_total+=psnr
            ssim_avg_total+=ssim

            print('psnr  '+str(psnr)+ ' ssim   ' +str(ssim))
        print(each+' psnr '+str(psnr_avg_seq/97))
        print(each+' ssim '+str(ssim_avg_seq/97))
    print('total psnr '+str(psnr_avg_total/(97*len(out_img_p1_dir))))
    print('total ssim '+str(ssim_avg_total/(97*len(out_img_p1_dir))))
def cal_vid4():

    seq_name_tuple = ('calendar','city','foliage','walk')
    GT_base_dir =  '/home/zhangyuantong/dataset/Vid4/GT/'
    pred_base_dir = '/home/zhangyuantong/code/MyOpenSource/STSR-OFR/output/vid4'
    total_psnr = 0
    total_ssim = 0
    cnt = 0
    for q in range(4):
        tag = seq_name_tuple[q]
        img_gt_dir_ls = sorted([os.path.join(GT_base_dir,tag,each) for each in os.listdir(os.path.join(GT_base_dir,tag))])
        img_pred_dir_ls = sorted([os.path.join(pred_base_dir,tag,each) for each in os.listdir(os.path.join(pred_base_dir,tag))])
        seq_psnr = 0
        seq_ssim = 0
        print("scene: %s "%(tag))
        for ix,data in enumerate(img_gt_dir_ls):
            print("%d / %d "%(ix,len(img_gt_dir_ls)))
            img_gt = cv2.imread(img_gt_dir_ls[ix], cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.
            img_restored = cv2.imread(
            img_pred_dir_ls[ix],
            cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
            psnr = calculate_psnr(
            img_gt * 255,
            img_restored * 255,
            crop_border = 0
            )
            ssim = calculate_ssim(
            img_gt * 255,
            img_restored * 255,
             crop_border = 0
            )
            total_psnr+=psnr
            total_ssim+=ssim
            seq_psnr+=psnr
            seq_ssim+=ssim
            cnt+=1
            print('psnr :',psnr,' ssim: ',ssim)
        print(" for scene %s avg psnr Y %f"%(tag,seq_psnr/len(img_gt_dir_ls)))
        print(" for scene %s avg ssim Y %f"%(tag,seq_ssim/len(img_gt_dir_ls)))
    print(" total avg psnr Y %f"%(total_psnr/cnt))
    print(" total avg ssim Y %f"%(total_ssim/cnt))
def cal_vimeo():
 

    GT_base_dir =  'xxx'
    pred_base_dir = 'xxx'
    seq_dir = 'xxx'
    total_psnr = 0
    read_all = True
    if read_all:
        test_list_1 = open(seq_dir+'/'+'fast_testset.txt').readlines()
        test_list_2 = open(seq_dir+'/'+'medium_testset.txt').readlines()
        test_list_3 = open(seq_dir+'/'+'slow_testset.txt').readlines()
        test_list = test_list_1+test_list_2+test_list_3
    img_pred_dir_ls = sorted([os.path.join(pred_base_dir,each.strip()) for each in test_list])
    total_psnr = 0
    total_ssim = 0
    total_len = 0

    for ix,p in enumerate(img_pred_dir_ls):
       
            this_d_gt = os.path.join(GT_base_dir,p.split('/')[-2],p.split('/')[-1])

            this_d = os.path.join(img_pred_dir_ls[ix])

            print(ix)
            print(p.split('/')[-2]+'/'+p.split('/')[-1])
            avg_psnr_7 = 0
            
            for i in range(7):
               

                img_gt = cv2.imread(this_d_gt+'/im'+str(i+1)+'.png').astype(
                np.float32) / 255.

                img_restored = cv2.imread(this_d+'/'+str(i)+'.png').astype(np.float32) / 255.
                img_gt = bgr2ycbcr(img_gt, y_only=True)
                img_restored = bgr2ycbcr(img_restored, y_only=True)

                psnr = calculate_psnr(
                img_gt * 255,
                img_restored * 255,
                 crop_border = 0
                )
                ssim = calculate_ssim(
                img_gt * 255,
                img_restored * 255,
                 crop_border = 0
                )
                avg_psnr_7+=psnr
                total_psnr+=psnr
                total_ssim+=ssim

                print('psnr  '+str(psnr)+ ' ssim   ' +str(ssim))
                # print(ssim)
                total_len+=1
            print('avg : ',avg_psnr_7/7 )

   
    print(" total avg psnr Y %f"%(total_psnr/total_len))
    print(" total avg ssim Y %f"%(total_ssim/total_len))

if __name__=='__main__':
    # specify imageset you want to test
    cal_vid4()
    # cal_REDS()
    
    # cal_vimeo()
