# Awesome Low-light Enhancement

If you have any problems, suggestions or improvements, please submit the issue or PR.

[Paper with Code](https://paperswithcode.com/task/low-light-image-enhancement)

---

## Contents

- [Datasets](#datasets)
- [Papers](#papers)

## Datasets

- VV, LIME, NPE-series, DICM, MEF
  - only low-light images without corresponding high-light ground truth
  - [Download](https://drive.google.com/drive/folders/0B_FjaR958nw_djVQanJqeEhUM1k?usp=sharing) (Thanks to [baidut](https://github.com/baidut/BIMEF) for collection)
- SID
  - [Homepage](http://cchen156.web.engr.illinois.edu/SID.html)
  - Download: [Sony part](https://storage.googleapis.com/isl-datasets/SID/Sony.zip) + [Fuji part](https://storage.googleapis.com/isl-datasets/SID/Fuji.zip) or [Baidu Drive](https://pan.baidu.com/s/1fk8EibhBe_M1qG0ax9LQZA)
- MIT-Adobe 5K
  - [Homepage](https://data.csail.mit.edu/graphics/fivek/)
  - [Download](https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar) (use only the output by Expert C)
- ExDARK
  - only low-light image without corresponding high-light ground truth
  - [Homepage](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset)
  - [Download](http://web.fsktm.um.edu.my/~cschan/source/CVIU/ExDark.zip)
- LOL
  - [Homepage](https://daooshee.github.io/BMVC2018website/)
  - Download: [Google Drive](https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB) or [Baidu Pan (Code: acp3)](https://pan.baidu.com/s/1ABMrDjBTeHIJGlOFIeP1IQ)
- LOL-Real (LOL v2):
  - [Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing) or [Baidu Pan (Code: l9xm)](https://pan.baidu.com/s/1U9ePTfeLlnEbr5dtI1tm5g)
  - both V1 & V2 can be found [here](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)
- SMID-DRV (dark raw video dataset)
  - [Homepage](https://github.com/cchen156/Seeing-Motion-in-the-Dark)
- SICE
  - [Homepage](https://github.com/csjcai/SICE)
  - Google Drive: [part1](https://goo.gl/gTGfLk) + [part2](https://goo.gl/ciV2C5)
  - Baudu Yun: [part1](https://pan.baidu.com/s/1kXotehL) + [part2](https://pan.baidu.com/s/1x1Dq9xef1dBTXXHcMjPAyA) + [part2 Label](https://pan.baidu.com/s/1zZR5xU92q7UwcCJq-_9xmQ)
- DeepUPE (not public)
- ELD
  - [Homepage](https://github.com/Vandermode/ELD)
  - Download: [Google Drive](https://drive.google.com/drive/folders/1CT2Ny9W9ArdSQaHNxC5hGwav9lZHoqJa?usp=sharing) or [Baidu Pan (Code: 0lby)](https://pan.baidu.com/s/11ksugpPH5uyDL-Z6S71Q5g)

- LSRW
  - Paper (R2RNet): [Arxiv](https://arxiv.org/abs/2106.14501) or [J Vis Commun Image R](https://www.sciencedirect.com/science/article/pii/S1047320322002322)
  - [Homepage (github project)](https://github.com/JianghaiSCU/R2RNet)
  - Download: [Baidu Pan (Code: wmrr)](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA)
- DARK FACE
  - joint with face detection
  - [Homepage](https://flyywh.github.io/CVPRW2019LowLight/)
  - Google Drive: [train&val](https://drive.google.com/file/d/10W3TDvEAlZfEt88hMxoEuRULr42bIV7s/view) + [test](https://drive.google.com/file/d/1MLeLUf8H4CqvWUoSn5TePqRWznueWvig/view?usp=sharing)
  - Baidu Pan: [train&val (Code: babu)](https://pan.baidu.com/s/1oSUwsq457eMMCnSSp7gGRQ) + [test (Code: 429h)](https://pan.baidu.com/s/1Fq0zyIJ0QPOcY3vqh427BQ)
- ACDC
  - 1006 nighttime images, joint with semantic segmentation
  - [Benchmark](https://acdc.vision.ee.ethz.ch/)
  - [Overview](https://acdc.vision.ee.ethz.ch/overview) & [Download](https://acdc.vision.ee.ethz.ch/download)
- SDSD
  - video dataset for supervised methods
  - [Homepage](https://github.com/dvlab-research/SDSD)
  - Download: [Google Drive](https://drive.google.com/drive/folders/1-fQGjzNcyVcBjo_3Us0yM5jDu0CKXXrV?usp=sharing) or [Baidu Pan (Code: zcrb)](https://pan.baidu.com/s/1CSNP_mAJQy1ZcHf5kXSrFQ)
- Dark Zurich
  - contains corresponding images of the same driving scenes at daytime, twilight and nighttime.
  - [homepage](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/)
- Real-LOL-Blur
  - joint with deblur
  - [Homepage](https://shangchenzhou.com/projects/LEDNet/)
  - [Download (Google Drive)](https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX)
- MSEC
  - each image contains **either over- or under-exposure** errors
  - [homepage](https://github.com/mahmoudnafifi/Exposure_Correction)
  - [Training](https://ln2.sync.com/dl/141f68cf0/mrt3jtm9-ywbdrvtw-avba76t4-w6fw8fzj) & [Validation](https://ln2.sync.com/dl/49a6738c0/3m3imxpe-w6eqiczn-vripaqcf-jpswtcfr) & [Test](https://ln2.sync.com/dl/098a6c5e0/cienw23w-usca2rgh-u5fxikex-q7vydzkp)
- LCD
  - each image contains **both over- and under-exposure** errors.
  - [homepage](https://hywang99.github.io/2022/07/09/lcdpnet/)
  - [download](https://drive.google.com/drive/folders/10Reaq-N0DiZiFpSrZ8j5g3g0EJes4JiS)
- GTA5
  - Synthetic GTA5 nighttime fog data [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570460.pdf)]
  - [download](https://www.dropbox.com/sh/gfw44ttcu5czrbg/AACr2GZWvAdwYPV0wgs7s00xa?dl=0)


## Papers

### Supervised Method

- **[SID]** Learning to See in the Dark (**CVPR 2018**) [[paper](http://cchen156.web.engr.illinois.edu/paper/18CVPR_SID.pdf)][[code](https://github.com/cchen156/Learning-to-See-in-the-Dark)][[code-pytorch](https://github.com/cydonia999/Learning_to_See_in_the_Dark_PyTorch)]

- **[Retinex-Net]** Deep Retinex Decomposition for Low-Light Enhancement (**BMVC 2018**) [[paper](https://arxiv.org/pdf/1808.04560)][[code](https://github.com/weichen582/RetinexNet)][[code-pytorch](https://github.com/FunkyKoki/RetinexNet_PyTorch)]

- **[HDRNet]** Deep Bilateral Learning for Real-Time Image Enhancement (**SIGGRAPH 2017**) [[paper](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf)][[code](https://github.com/google/hdrnet)]

- **[DeepUPE]** Underexposed Photo Enhancement using Deep Illumination Estimation (**CVPR 2019**) [[paper](https://drive.google.com/file/d/1CCd0NVEy0yM2ulcrx44B1bRPDmyrgNYH/view)][[code (only test)](https://github.com/wangruixing/DeepUPE)]

- **[CWAN]** Color-wise Attention Network for Low-light Image Enhancement [[paper](https://arxiv.org/pdf/1911.08681)]

- **[SMID]** Seeing Motion in the Dark (**ICCV 2019**) [[homepage](https://vladlen.info/publications/seeing-motion-dark/)][[code-tensorflow](https://github.com/cchen156/Seeing-Motion-in-the-Dark)]
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

- **[DCGANs]** Deep Learning for Robust end-to-end Tone Mapping (**BMVC 2019**) [[paper](https://bmvc2019.org/wp-content/uploads/papers/0849-paper.pdf)]

- **[RJI]** Robust Joint Image Reconstruction from Color and Monochrome Cameras (**BMVC 2019**) [[paper](https://bmvc2019.org/wp-content/uploads/papers/0754-paper.pdf)]

- **[FIDE]** Learning to Restore Low-Light Images via Decomposition-and-Enhancement (**CVPR 2020**) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Learning_to_Restore_Low-Light_Images_via_Decomposition-and-Enhancement_CVPR_2020_paper.pdf)]

- **[Pb-NFM]** A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising (**CVPR 2020**, paper for [ELD dataset](#ELD)) [[paper](https://arxiv.org/abs/2003.12751)][[Code](https://github.com/Vandermode/ELD)]

- **[DALE]** DALE : Dark Region-Aware Low-light Image Enhancement (**BMVC 2020**) [[paper](https://www.bmvc2020-conference.com/assets/papers/1025.pdf)]

- **[LLPackNet]** Towards Fast and Light-Weight Restoration of Dark Images (**BMVC 2020**) [[[paper](https://www.bmvc2020-conference.com/assets/papers/0145.pdf)] [[code](https://github.com/MohitLamba94/LLPackNet)]

- **[StableLLVE]** Learning Temporal Consistency for Low Light Video Enhancement
from Single Images (**CVPR 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Learning_Temporal_Consistency_for_Low_Light_Video_Enhancement_From_Single_CVPR_2021_paper.html)][[Code](StableLLVE)]

- **[UTVNet]** Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement (**ICCV 2021**) [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Adaptive_Unfolding_Total_Variation_Network_for_Low-Light_Image_Enhancement_ICCV_2021_paper.html)][[Code](https://github.com/CharlieZCJ/UTVNet)]

- **[SNRA]** SNR-Aware Low-light Image Enhancement (**CVPR 2022**) [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.html)][[code](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance)]

- **[URetinex]** URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light Image Enhancement (**CVPR 2022**) [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html)]

- **[DCC-Net]** Deep Color Consistent Network for Low-Light Image Enhancement [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Deep_Color_Consistent_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf)]

- **[LEDNet]** LEDNet: Joint Low-light Enhancement and Deblurring in the Dark (**ECCV 2022**) [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1331_ECCV_2022_paper.php)][[code](https://github.com/sczhou/LEDNet)]

- **[LCDPNet]** Local Color Distributions Prior for Image Enhancement (**ECCV 2022**) [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2257_ECCV_2022_paper.php)][[homepage](https://hywang99.github.io/2022/07/09/lcdpnet/)]

- **[IAT]** You Only Need 90K Parameters to Adapt Light: a Light Weight Transformer for Image Enhancement and Exposure Correction (**BMVC 2022**) [[project](https://bmvc2022.mpi-inf.mpg.de/238/)][[paper](https://bmvc2022.mpi-inf.mpg.de/0238.pdf)][[code](https://github.com/cuiziteng/Illumination-Adaptive-Transformer)]

- **[CFA-LLVE]** Low Light Video Enhancement by Learning on Static Videos with Cross-Frame Attention (**BMVC 2022**) [[project](https://bmvc2022.mpi-inf.mpg.de/743/)][[paper](https://bmvc2022.mpi-inf.mpg.de/0743.pdf)]

## Semi-Superised and Unsupervised Method

- **[LIME]** LIME: Low-light Image Enhancement via Illumination Map Estimation (**T-IP**) [[paper](https://ieeexplore.ieee.org/document/7782813)] [[code-matlab](https://github.com/Sy-Zhang/LIME)]

- **[EnlightenGAN]** EnlightenGAN: Deep Light Enhancement without Paired Supervision [[paper](https://arxiv.org/pdf/1906.06972)][[code](https://github.com/TAMU-VITA/EnlightenGAN)]

- **[Zero-DCE]** Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (**CVPR 2020**) [[paper](https://arxiv.org/abs/2001.06826v2)][[homepage](https://li-chongyi.github.io/Proj_Zero-DCE.html)][[code](https://github.com/Li-Chongyi/Zero-DCE)]

- **[DRBN]** From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (**CVPR 2020**) [[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_From_Fidelity_to_Perceptual_Quality_A_Semi-Supervised_Approach_for_Low-Light_CVPR_2020_paper.pdf)][[Code](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)]

- **[RUAS]** Retinex-inspired Unrolling with Cooperative Prior Architecture Search for Low-light Image Enhancement (**CVPR 2021**) [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.html)][[Code](https://github.com/KarelZhang/RUAS)]
- **[SSIE]** Self-supervised Image Enhancement Network: Training with Low Light Images Only (**Arxiv**) [[paper](https://arxiv.org/abs/2002.11300)][[code](https://github.com/hitzhangyu/Self-supervised-Image-Enhancement-Network-Training-With-Low-Light-Images-Only)]

- **[SCI]** Toward Fast, Flexible, and Robust Low-Light Image Enhancement (**CVPR 2022**) [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.html)][[Code](https://github.com/vis-opt-group/SCI)]
- **[LES]** Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression (**ECCV 2022**) [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/982_ECCV_2022_paper.php)][[code](https://github.com/jinyeying/night-enhancement)]

- **[UDCL-Transformer]** Unsupervised Low Light Image Enhancement Transformer Based on Dual Contrastive Learning (**BMVC 2022**) [[project](https://bmvc2022.mpi-inf.mpg.de/373/)][[paper](https://bmvc2022.mpi-inf.mpg.de/0373.pdf)][[code](https://github.com/KaedeKK/UDCL-Transformer)]
