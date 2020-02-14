# Awesome Low-light Enhancement

If you have any problems, suggestions or improvements, please submit the issue or PR.

---

## Contents

- [Datasets](#datasets)
- [Papers](#papers)

## Datasets

- VV, LIME, NPE-series, DICM, MEF
  - only low-light images without corresponding high-light ground truth
  - [Download](https://drive.google.com/drive/folders/0B_FjaR958nw_djVQanJqeEhUM1k?usp=sharing)
  - Thanks to [baidut](https://github.com/baidut/BIMEF) for collection
- SID
  - [Homepage](http://cchen156.web.engr.illinois.edu/SID.html)
  - Download: [Sony part](https://storage.googleapis.com/isl-datasets/SID/Sony.zip) + [Fuji part](https://storage.googleapis.com/isl-datasets/SID/Fuji.zip) or [Baidu Drive](https://pan.baidu.com/s/1fk8EibhBe_M1qG0ax9LQZA)
- MIT-Adobe FiveK
  - [Homepage](https://data.csail.mit.edu/graphics/fivek/)
  - [Download](https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar) (use only the output by Expert C)
- ExDARK
  - only low-light image without corresponding high-light ground truth
  - [Homepage](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset)
  - [Download](http://web.fsktm.um.edu.my/~cschan/source/CVIU/ExDark.zip)
- LOL
  - [Homepage](https://daooshee.github.io/BMVC2018website/)
  - Download: [Google Drive](https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB) or [Baidu Pan (Code: acp3)](https://pan.baidu.com/s/1ABMrDjBTeHIJGlOFIeP1IQ)
- SICE
  - [Homepage](https://github.com/csjcai/SICE)
  - Google Drive: [part1](https://goo.gl/gTGfLk) + [part2](https://goo.gl/ciV2C5)
  - Baudu Yun: [part1](https://pan.baidu.com/s/1kXotehL) + [part2](https://pan.baidu.com/s/1x1Dq9xef1dBTXXHcMjPAyA) + [part2 Label](https://pan.baidu.com/s/1zZR5xU92q7UwcCJq-_9xmQ)
- DeepUPE (not public)

## Papers

### Supervised Method

- **[LIME]** LIME: Low-light Image Enhancement via Illumination Map Estimation (**T-IP**) [[paper](https://ieeexplore.ieee.org/document/7782813)] [[code-matlab](https://github.com/Sy-Zhang/LIME)]

- **[SID]** Learning to See in the Dark (**CVPR 2018**) [[paper](http://cchen156.web.engr.illinois.edu/paper/18CVPR_SID.pdf)][[code](https://github.com/cchen156/Learning-to-See-in-the-Dark)][[code-pytorch](https://github.com/cydonia999/Learning_to_See_in_the_Dark_PyTorch)]

- **[Retinex-Net]** Deep Retinex Decomposition for Low-Light Enhancement (**BMVC 2018**) [[paper](https://arxiv.org/pdf/1808.04560)][[code](https://github.com/weichen582/RetinexNet)][[code-pytorch](https://github.com/FunkyKoki/RetinexNet_PyTorch)]

- **[HDRNet]** Deep Bilateral Learning for Real-Time Image Enhancement (**SIGGRAPH 2017**) [[paper](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf)][[code](https://github.com/google/hdrnet)]

- **[DeepUPE]** Underexposed Photo Enhancement using Deep Illumination Estimation (**CVPR 2019**) [[paper](https://drive.google.com/file/d/1CCd0NVEy0yM2ulcrx44B1bRPDmyrgNYH/view)][[code (only test)](https://github.com/wangruixing/DeepUPE)]

- **[CWAN]** Color-wise Attention Network for Low-light Image Enhancement [[paper](https://arxiv.org/pdf/1911.08681)]

- **[MSR-net]** MSR-net:Low-light Image Enhancement Using Deep Convolutional Network [[paper](https://arxiv.org/pdf/1711.02488)]

- **[GLADNet]** GLADNet: Low-Light Enhancement Network with Global Awareness (**FG 2018**) [[paper](https://ieeexplore.ieee.org/document/8373911)][[code](https://github.com/weichen582/GLADNet)]

- **[LIE-GP]** Low-light image enhancement using Gaussian Process for features retrieval (**SPIC**) [[paper](http://cs-chan.com/doc/SPIC2019.pdf)][[code](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/SPIC)]

- **[MBLLEN]** MBLLEN: Low-light Image/Video Enhancement Using CNNs (**BMVC 2018**) [[paper](http://bmvc2018.org/contents/papers/0700.pdf)][[code](https://github.com/Lvfeifan/MBLLEN)]

- **[LLNet]** LLNet: Low-light Image Enhancement with Deep Learning ((**PR**)) [[paper](https://arxiv.org/pdf/1511.03995)][[code](https://github.com/kglore/llnet_color)]

- **[KinD]** Kindling the Darkness: A Practical Low-light Image Enhancer (**ACM MM 2019**) [[paper](https://arxiv.org/pdf/1905.04161)] [[code](https://github.com/zhangyhuaee/KinD)]

- **[LL-RefineNet]** Deep Refinement Network for Natural Low-Light Image Enhancement in Symmetric Pathways (**Symmetry**) [[paper](https://www.mdpi.com/2073-8994/10/10/491/pdf)]

- **[LLCNN]**  LLCNN: A convolutional neural network for low-light image enhancement (**VCIP 2017**) [[paper](https://ieeexplore.ieee.org/abstract/document/8305143)][[code](https://github.com/BestJuly/LLCNN)]

- **[LLEDHN]** Low-Light Image Enhancement via a Deep Hybrid Network (**T-IP**) [[paper](https://ieeexplore.ieee.org/document/8692732)][[code](https://drive.google.com/file/d/1WYQd5z9NXW-IOWLSH3w70t3XnLUAHnAZ/view?usp=sharing)]

- **[SICE]** Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images (**T-IP**) [[paper](http://www4.comp.polyu.edu.hk/~cslzhang/paper/SICE.pdf)][[code](https://github.com/csjcai/SICE)]

- **[ALIE]** Attention-guided Low-light Image Enhancement [[paper](https://arxiv.org/pdf/1908.00682)] [[homepage](http://phi-ai.org/project/AgLLNet/default.htm)]

## Unsupervised Method

- **[EnlightenGAN]** EnlightenGAN: Deep Light Enhancement without Paired Supervision [[paper](https://arxiv.org/pdf/1906.06972)][[code](https://github.com/TAMU-VITA/EnlightenGAN)]

- **[Zero-DCE]** Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement [[paper](https://arxiv.org/pdf/2001.06826)][[code](https://github.com/Li-Chongyi/Zero-DCE)]



