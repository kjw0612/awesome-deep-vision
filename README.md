# Awesome Deep Vision
A curated list of deep learning resources for computer vision, inspired by [awesome-php](https://github.com/ziadoz/awesome-php) and [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).


CVPR 2015 Papers to be Added Soon!

## Contributing
Please feel free to [pull requests](https://github.com/kjw0612/awesome-deep-vision/pulls) or email jiwon@alum.mit.edu to add links.

## Sharing
+ [Share on Twitter](http://twitter.com/home?status=https://github.com/kjw0612/awesome-deep-vision%0ADeep Learning Resources for Computer Vision)
+ [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://github.com/kjw0612/awesome-deep-vision)
+ [Share on Google Plus](https://plus.google.com/share?url=https://github.com/kjw0612/awesome-deep-vision)
+ [Share on LinkedIn](https://www.linkedin.com/shareArticle?mini=true&url=https://github.com/kjw0612/awesome-deep-vision&title=Awesome%20Deep%20Vision&summary=&source=)

## Table of Contents
 - [Papers](#papers)
  - [ImageNet Classification](#imagenet-classification)
  - [Image Captioning](#image-captioning)
  - [Low-Level Vision](#low-level-vision)
  - [Edge Detection](#edge-detection)
  - [Semantic Segmentation](#semantic-segmentation)
 - [Courses](#courses)
 - [Software](#software)
 - [Tutorials](#tutorials)
 
## Papers

### ImageNet Classification
  * Microsoft (PReLu/Weight Initialization) [[Paper]](http://arxiv.org/pdf/1502.01852v1)
    * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, arXiv:1502.01852.
  * Batch Normalization [[Paper]](http://arxiv.org/pdf/1502.03167v3)
    * Sergey Ioffe, Christian Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, arXiv:1502.03167.
  * GoogLeNet [[Paper]](http://arxiv.org/pdf/1409.4842v1)
    * Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich, CVPR 2015. 
  * VGG-Net [[Web]](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) [[Paper]](http://arxiv.org/pdf/1409.1556)
   * Karen Simonyan and Andrew Zisserman, Very Deep Convolutional Networks for Large-Scale Visual Recognition, ICLR 2015.
  * AlexNet [[Paper]](http://books.nips.cc/papers/files/nips25/NIPS2012_0534.pdf)
    * Krizhevsky, A., Sutskever, I. and Hinton, G. E, ImageNet Classification with Deep Convolutional Neural Networks
NIPS 2012.

### Image Captioning 
   * Baidu/UCLA [[Paper]](http://arxiv.org/pdf/1410.1090v1)
      * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Alan L. Yuille, Explain Images with Multimodal Recurrent Neural Networks, arXiv:1410.1090 (2014).
   * Toronto [[Paper]](http://arxiv.org/pdf/1411.2539v1)
      * Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel, Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models, arXiv:1411.2539 (2014).
   * Berkeley [[Paper]](http://arxiv.org/pdf/1411.4389v3)
      * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, Trevor Darrell, Long-term Recurrent Convolutional Networks for Visual Recognition and Description, arXiv:1411.4389 (2014).
   * Google [[Paper]](http://arxiv.org/pdf/1411.4555v2)
      * Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan, Show and Tell: A Neural Image Caption Generator, arXiv:1411.4555 (2014). 
   * Stanford [[Web]](http://cs.stanford.edu/people/karpathy/deepimagesent/) [[Paper]](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
      * Andrej Karpathy, Li Fei-Fei, Deep Visual-Semantic Alignments for Generating Image Description, CVPR (2015).
   * UML/UT [[Paper]](http://arxiv.org/pdf/1412.4729v3)
      * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, Kate Saenko, Translating Videos to Natural Language Using Deep Recurrent Neural Networks, NAACL-HLT 2015. 
   * Microsoft/CMU [[Paper]](http://arxiv.org/pdf/1411.5654v1)
      * Xinlei Chen, C. Lawrence Zitnick, Learning a Recurrent Visual Representation for Image Caption Generation, arXiv:1411.5654.
   * Microsoft [[Paper]](http://arxiv.org/pdf/1411.4952v3)
      * Hao Fang, Saurabh Gupta, Forrest Iandola, Rupesh Srivastava, Li Deng, Piotr Dollár, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C. Platt, C. Lawrence Zitnick, Geoffrey Zweig, From Captions to Visual Concepts and Back, CVPR 2015. 

### Low-Level Vision
 * Optical Flow (FlowNet) [[Paper]](http://arxiv.org/pdf/1504.06852v2)
  * Philipp Fischer, Alexey Dosovitskiy, Eddy Ilg, Philip Häusser, Caner Hazırbaş, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox, FlowNet: Learning Optical Flow with Convolutional Networks, arXiv:1504.06852.
 * Super-Resolution (SRCNN) [[Web]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) [[Paper-ECCV14]](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf) [[Paper-arXiv15]](http://arxiv.org/pdf/1501.00092v1.pdf)
    * Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, Learning a Deep Convolutional Network for Image Super-Resolution, in ECCV 2014
    * Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks, arXiv:1501.00092 (2015)
 * Compression Artifacts Reduction [[Paper-arXiv15]](http://arxiv.org/pdf/1504.06993v1)
   * Chao Dong, Yubin Deng, Chen Change Loy, Xiaoou Tang, Compression Artifacts Reduction by a Deep Convolutional Network, arXiv:1504.06993
 * Non-Uniform Motion Blur Removal [[Paper]](http://arxiv.org/pdf/1503.00593v3)
  * Jian Sun, Wenfei Cao, Zongben Xu, Jean Ponce, Learning a Convolutional Neural Network for Non-uniform Motion Blur Removal, CVPR 2015. 
 * Image Deconvolution [[Web]](http://lxu.me/projects/dcnn/) [[Paper]](http://lxu.me/mypapers/dcnn_nips14.pdf)
  *  Li Xu, Jimmy SJ. Ren, Ce Liu, Jiaya Jia, "Deep Convolutional Neural Network for Image Deconvolution" Advances in Neural Information Processing Systems (NIPS), 2014.

### Edge Detection
 * Holistically-Nested Edge Detection [[Paper]](http://arxiv.org/pdf/1504.06375v1)
  * Saining Xie, Zhuowen Tu, Holistically-Nested Edge Detection, arXiv:1504.06375. 
 * DeepEdge [[Paper]](http://arxiv.org/pdf/1412.1123v3)
  * Gedas Bertasius, Jianbo Shi, Lorenzo Torresani, DeepEdge: A Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection, CVPR 2015.
 * DeepContour [[Paper]](http://mc.eistar.net/UpLoadFiles/Papers/DeepContour_cvpr15.pdf)
  * Wei Shen, Xinggang Wang, Yan Wang, Xiang Bai, Zhijiang Zhang, DeepContour: A Deep Convolutional Feature Learned by Positive-Sharing Loss for Contour Detection, CVPR 2015.

### Semantic Segmentation
  * Learning Hierarchical Features for Scene Labeling [[Paper-ICML12]](http://yann.lecun.com/exdb/publis/pdf/farabet-icml-12.pdf) [[Paper-PAMI13]](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
   * Clement Farabet, Camille Couprie, Laurent Najman, Yann LeCun, Scene Parsing with Multiscale Feature Learning, Purity Trees, and Optimal Covers, ICML, 2012.
   * Clement Farabet, Camille Couprie, Laurent Najman, Yann LeCun, Learning Hierarchical Features for Scene Labeling, PAMI, 2013.
  * R-CNN [[Paper-CVPR14]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) [[Paper-arXiv14]](http://arxiv.org/pdf/1311.2524v5)
   * Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik, Rich feature hierarchies for accurate object detection and semantic segmentation, CVPR, 2014.
  * Fully Convolutional Networks for Semantic Segmentation [[Paper-CVPR15]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) [[Paper-arXiv15]](http://arxiv.org/pdf/1411.4038v2)
   * Jonathan Long, Evan Shelhamer, Trevor Darrell, Fully Convolutional Networks for Semantic Segmentation, CVPR, 2015.
  * Conditional Random Fields as Recurrent Neural Networks [[Paper]](http://arxiv.org/pdf/1502.03240v2)
   * Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, Philip H. S. Torr, Conditional Random Fields as Recurrent Neural Networks, arXiv:1502.03240
  * BoxSup [[Paper]](http://arxiv.org/pdf/1503.01640v2)
   * Jifeng Dai, Kaiming He, Jian Sun, BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation, arXiv:1503.01640

## Courses
 * [Stanford] [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
 * [CUHK] [ELEG 5040: Advanced Topics in Signal Processing(Introduction to Deep Learning)](https://piazza.com/cuhk.edu.hk/spring2015/eleg5040/home)

## Software 
### Framework
 * Torch7: Deep learning library in Lua, used by Facebook and Google Deepmind [[Web]](http://torch.ch/)
 * Caffe: Deep learning framework by the BVLC [[Web]](http://caffe.berkeleyvision.org/)
 * MatConvNet: CNNs for MATLAB [[Web]](http://www.vlfeat.org/matconvnet/)

### Applications
 * Adversarial Training 
  * Code and hyperparameters for the paper "Generative Adversarial Networks" [[Web]](https://github.com/goodfeli/adversarial)
 * Understanding and Visualizing
  * Source code for "Understanding Deep Image Representations by Inverting Them", CVPR 2015. [[Web]](https://github.com/aravindhm/deep-goggle)
 * Semenatic Segmentation
 
## Tutorials
  * [CVPR 2014] [Tutorial on Deep Learning in Computer Vision](https://sites.google.com/site/deeplearningcvpr2014/)
  * [CVPR 2015] [Applied Deep Learning for Computer Vision with Torch](http://torch.ch/docs/cvpr15.html)

Maintainers - [Jiwon Kim](http://github.com/kjw0612), [Heesoo Myeong](https://github.com/hmyeong), [Myungsub Choi](http://github.com/myungsub), [JanghoonChoi](https://github.com/JanghoonChoi), [Jung Kwon Lee](http://github.com/deruci)
