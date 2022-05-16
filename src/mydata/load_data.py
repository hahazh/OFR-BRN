from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

import random
from PIL import Image
from torchvision import transforms
import sys
import torch
import math
import  mydata.core_bicubic as core
def bicubic_downsample(x, scale=4):
    C, T, H, W = x.size()
    ret_ls = []
    for i in range(T):
        img = x[:,i,:,:]
        img = core.imresize(img,sizes=(H//scale,W//scale))

        ret_ls.append(img.unsqueeze(1))
    
    ret = torch.cat(ret_ls,dim=1)

    return ret

def get_loader_vimeo( data_root, test_file_name, batch_size, shuffle, num_workers):
   
    data_augmentation = False
    file_list = test_file_name
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Vimeo_SepTuplet_ST(data_root, 4, data_augmentation, file_list, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

class Vimeo_SepTuplet_ST(Dataset):  # 
    def __init__(self, image_dir, scale, data_augmentation, file_list, transform):
        super(Vimeo_SepTuplet_ST, self).__init__()
      
        alist = [line.rstrip() for line in open(os.path.join(image_dir,
                                                             file_list))]  # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
        self.image_filenames = [os.path.join(image_dir, x) for x in alist]  # get image path list
        
        self.scale = scale

        self.transform = transform  # To_tensor
        self.data_augmentation = data_augmentation  # flip and rotate
        self.LRindex = [0, 2, 4, 6]
        self.mindex = [1,3,5]



    def load_img(self, image_path, scale):
        HR = []
        for img_num in range(7):
            img_gt = Image.open(os.path.join(image_path, 'im{}.png'.format(img_num + 1))).convert('RGB')

            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]


        return HR

    def __getitem__(self, index):


        GT = self.load_img(self.image_filenames[index], self.scale)

        GT = np.asarray(GT)  # numpy, [T,H,W,C], stack with temporal dimension
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]

        LR = bicubic_downsample(GT,self.scale)
        LR_in = LR[:, self.LRindex]
        LR_mid = LR[:, self.mindex ]

        LR_in = LR_in.permute(1, 0, 2, 3)
        LR_mid = LR_mid.permute(1, 0, 2, 3)
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]
       
        GT_long = torch.cat((GT,torch.flip(GT[:-1],dims = [0]) ),dim=0) 
        LR_long = torch.cat((LR_in,torch.flip(LR_in[:-1],dims = [0]) ),dim=0)
        LR_mid_long = torch.cat((LR_mid,torch.flip(LR_mid,dims = [0]) ),dim=0)
        GT,LR_in,LR_mid = GT_long,LR_long,LR_mid_long
             
        
          
        return LR_in,self.image_filenames[index]
       

    def __len__(self):
        return len(self.image_filenames)  


def get_vid4_loader(data_root,scale,seq_name,seq_length, batch_size, num_workers):
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Vid4(data_root, seq_name,scale, transform,seq_length,mode='overlap')
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


class Vid4(Dataset):  
    def __init__(self, data_dir,seq_name, scale, transform, seq_length=14,mode='mirror'):
        super(Vid4, self).__init__()
        self.mode = mode
        self.vid4_dir = data_dir + seq_name
        self.total_len = len(os.listdir(self.vid4_dir))
        self.seq_length = seq_length
        self.alist = self.split_seq() 

        self.scale = scale

        self.transform = transform 
        self.LRindex = [int(i) for i in range(seq_length) if i%2==0]
        
        self.HRindex = [0, 2, 3, 4, 6]
    def split_seq(self):
       
        return  [range(len(os.listdir(self.vid4_dir)))]
    def load_img(self, image_path):
        HR = []
        for img_num in image_path:
            img_gt = Image.open(os.path.join(self.vid4_dir, str(img_num).zfill(8)+'.png')).convert('RGB')
            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
        
        return HR

    def __getitem__(self, index):

        GT = self.load_img(self.alist[index])
        GT = np.asarray(GT)  # numpy, [T,H,W,C], stack with temporal dimension
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]

        if h%32!=0:
            padded_h1 = (32-h%32)//2
            padded_h2 = (32-h%32)-padded_h1
            h = h+(32-h%32)
       
        else:
            padded_h1,padded_h2 =0,0
        if w%32!=0:
            padded_w1 = (32-w%32)//2
            padded_w2 = (32-w%32)-padded_w1
            w = w+(32-w%32)
        else:
            padded_w1,padded_w2 =0,0
     
        crop_shape = [padded_h1,h-padded_h2,padded_w1,w-padded_w2]
        if self.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0,0),(padded_h1,padded_h2),(padded_w1,padded_w2),(0,0)), mode='reflect')

        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)
        if self.transform:
            GT = self.transform(GT)  
        GT = GT.view(c, t, h, w)  

        LR = bicubic_downsample(GT[:, self.LRindex],self.scale)
        LR = LR.permute(1, 0, 2, 3)
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]



        return GT,LR,self.alist[index],crop_shape

    def __len__(self):
        return len(self.alist)  # total video number. not image number



def get_REDS_loader(data_root,scale,seq_length, batch_size, num_workers,shuffle=True,mode='overlap'):

    transform = transforms.Compose([transforms.ToTensor()])
  
    reds = REDS(data_root,scale,transform,seq_length,mode)
    return DataLoader(reds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class REDS(Dataset):  # load train dataset
    def __init__(self, data_dir, scale, transform, seq_length=14,mode='mirror'):
        super(REDS, self).__init__()
        self.mode = mode

       
       
        self.reds_seq_dir = sorted([os.path.join(data_dir,each.strip()) for each in open(os.path.join(data_dir,'test_list.txt')).readlines()])
        # length of reds
        self.total_len = 100
        self.seq_length = seq_length
        self.alist = self.split_seq() # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
        self.total_list = self.generate_total_list()

        self.scale = scale

        self.transform = transform  # To_tensor
        self.LRindex = [int(i) for i in range(seq_length) if i%2==0]
        self.mindex = [int(i) for i in range(seq_length) if i%2!=0]
        
        self.HRindex = [0, 2, 3, 4, 6]

    def split_seq(self):
        ret_ls = []
        append_ix = reversed([ix for ix in range(self.total_len-(self.seq_length-self.total_len%self.seq_length)-1,self.total_len-1)])
        sub_num = math.ceil(self.total_len/self.seq_length)
        if self.mode=='mirror':

            for seq_ix in range(sub_num):
                tmp_ix_ls = []
                if seq_ix==sub_num-1 :
                    for sub_ix in range(self.total_len%self.seq_length):
                        tmp_ix_ls.append(seq_ix*self.seq_length+sub_ix-1)
                    tmp_ix_ls+=append_ix
                else:
                    for sub_ix in range(self.seq_length):
                        tmp_ix_ls.append(seq_ix*self.seq_length+sub_ix)
                ret_ls.append(tmp_ix_ls)
        elif self.mode=='overlap':
            for seq_ix in range(sub_num):
                tmp_ix_ls = []
                if seq_ix==sub_num-1:
                    tmp_ix_ls = [int(i) for i in range(self.total_len-self.seq_length,self.total_len)]

                else:
                    for sub_ix in range(self.seq_length):
                        tmp_ix_ls.append(seq_ix*self.seq_length+sub_ix)
                ret_ls.append(tmp_ix_ls)
        return ret_ls
    def generate_total_list(self):
        total_ls = []
        for f_dir in self.reds_seq_dir:
            for s_dir in self.alist:
                tmp_ls  = []
                for im_d in s_dir:
                    im_d = str(im_d).zfill(8)+'.png'
                    this_dir = os.path.join(f_dir,im_d)
                    tmp_ls.append(this_dir)
       
                total_ls.append(tmp_ls)
            

        return total_ls

    def load_img(self, image_path):
      
        HR = []
        for img_name in image_path:
         
            img_gt = Image.open(img_name).convert('RGB')
            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
        H, W = HR[0].shape[:2]

        return HR

    def __getitem__(self, index):

     
        GT = self.load_img(self.total_list[index])
        GT = np.asarray(GT)  # numpy, [T,H,W,C], stack with temporal dimension
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
       

        if h%32!=0:
            padded_h1 = (32-h%32)//2
            padded_h2 = (32-h%32)-padded_h1
            h = h+(32-h%32)
    
        else:
            padded_h1,padded_h2 =0,0
        if w%32!=0:
            padded_w1 = (32-w%32)//2
            padded_w2 = (32-w%32)-padded_w1
            w = w+(32-w%32)
        else:
            padded_w1,padded_w2 =0,0
        padded_h1,padded_h2,padded_w1,padded_w2 =  padded_h1+16,padded_h2+16,padded_w1+16,padded_w2+16
        h,w = h+32,w+32
        crop_shape = [padded_h1,h-padded_h2,padded_w1,w-padded_w2]
        if self.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0,0),(padded_h1,padded_h2),(padded_w1,padded_w2),(0,0)), mode='reflect')

        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]
     
        LR = bicubic_downsample(GT,self.scale)
        LR_in =  LR[:, self.LRindex]
        LR_m = LR[:,self.mindex]
        LR_in =  LR_in.permute(1, 0, 2, 3)
        LR_m = LR_m.permute(1, 0, 2, 3)
      
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]

        
        return GT,LR_in,self.total_list[index],crop_shape
       

    def __len__(self):
        return len(self.total_list)  # total video number. not image number
