# Introduction

This is the implementation of  paper "Optical-Flow-Reuse-Based Bidirectional Recurrence
Network for Space-Time Video Super-Resolution".

# Pre-trained models

[BaiduCloud](https://pan.baidu.com/s/13-TYbvoFh7OmLtRY7uduWw)

password: opd3 

[dropbox](https://www.dropbox.com/scl/fi/txhjl4acez26b0rbfd9ak/ofr-brn.pth?rlkey=9lsz0lur89ewadmb26z242z30&dl=0)

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
or you may directly get all evaluation results in src/evaluation_results
# Run a demo





```
cd src

python demo.py
```

# Other STVSR works
We have conducted a series of video spatiotemporal super-resolution-related works, which include not only OFR-BRN but also:

Yuantong Zhang, Huairui Wang, Zhenzhong Chen: Controllable Space-Time Video Super-Resolution via Enhanced Bidirectional Flow Warping. VCIP 2022
Yuantong Zhang, Daiqin Yang, Zhenzhong Chen, Wenpeng Ding: Continuous Space-Time Video Super-Resolution Utilizing Long-Range Temporal Information. CoRR abs/2302.13256 (2023)

# Acknowledgment
Our code is built on

 [Zooming-Slow-Mo-CVPR-2020](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)

 [open-mmlab](https://github.com/open-mmlab)

 [bicubic_pytorch](https://github.com/sanghyun-son/bicubic_pytorch)

 [FLAVR](https://github.com/tarun005/FLAVR)

 [RIFE](https://github.com/hzwer/arXiv2020-RIFE)
 
 We thank the authors for sharing their codes!
