# Introduction

This is the implementation of  paper "Optical-Flow-Reuse-Based Bidirectional Recurrence
Network for Space-Time Video Super-Resolution".

# Pre-trained models

[BaiduCloud](https://pan.baidu.com/s/1GjE-HsQwKIwEZl0V5doZOQ)

password: 0wm8 

# Environment
We are good in the environment:

python 3.7

CUDA 9.1

Pytorch 1.5.0

# Generate image
To test on vimeo, 

```
cd src

python test_vimeo.py --datapath VIMEOPATH --outputpath  OUTPUTPATH --weight PATHTOWEIGHT
```

To test on REDS, 

```
cd src

python test_reds.py --datapath REDSPATH --outputpath  OUTPUTPATH --weight PATHTOWEIGHT
```

To test on VID4, 

```
cd src

python test_vid4.py --datapath VID4PATH --outputpath  OUTPUTPATH --weight PATHTOWEIGHT
```
# Calculate criteria
you should specify the GT path and output path first, and run:
```
cd src

python eval.py
```
or you may directly get all results in:

 src/evaluation_results
# Acknowledgment
Our code is built on

 [Zooming-Slow-Mo-CVPR-2020](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)

 [open-mmlab](https://github.com/open-mmlab)

 [bicubic_pytorch](https://github.com/sanghyun-son/bicubic_pytorch)

 [FLAVR](https://github.com/tarun005/FLAVR)

 [RIFE](https://github.com/hzwer/arXiv2020-RIFE)
 
 We thank the authors for sharing their codes!
