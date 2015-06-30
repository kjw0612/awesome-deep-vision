# Awesome Deep Vision
A curated list of deep learning resources for computer vision, inspired by [awesome-php](https://github.com/ziadoz/awesome-php) and [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

Maintainers - [Jiwon Kim](http://github.com/kjw0612), [Heesoo Myeong](https://github.com/hmyeong), [Myungsub Choi](http://github.com/myungsub), [JanghoonChoi](https://github.com/JanghoonChoi), [Jung Kwon Lee](http://github.com/deruci)

## Contributing
Please feel free to [pull requests](https://github.com/kjw0612/awesome-deep-vision/pulls) or email jiwon@alum.mit.edu to add links.

## Sharing
+ [Share on Twitter](http://twitter.com/home?status=http://jiwonkim.org/awesome-deep-vision%0ADeep Learning Resources for Computer Vision)
+ [Share on Facebook](http://www.facebook.com/sharer/sharer.php?u=https://jiwonkim.org/awesome-deep-vision)
+ [Share on Google Plus](http://plus.google.com/share?url=https://jiwonkim.org/awesome-deep-vision)
+ [Share on LinkedIn](http://www.linkedin.com/shareArticle?mini=true&url=https://jiwonkim.org/awesome-deep-vision&title=Awesome%20Deep%20Vision&summary=&source=)

## Table of Contents
 - [Papers](#papers)
  - [ImageNet Classification](#imagenet-classification)
  - [Image Captioning](#image-captioning)
  - [Low-Level Vision](#low-level-vision)
  - [Edge Detection](#edge-detection)
  - [Semantic Segmentation](#semantic-segmentation)
  - [Visual Attention and Saliency](#visual-attention-and-saliency)
  - [Object Recognition](#object-recognition)
  - [Understanding CNN](#understanding-cnn)
  - [Other Topics](#other-topics)
 - [Courses](#courses)
 - [Books](#books)
 - [Videos](#videos)
 - [Software](#software)
  - [Framework](#framework)
  - [Applications](#applications)
 - [Tutorials](#tutorials)
 - [Blogs](#blogs)
 
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
 * Super-Resolution (SRCNN) [[Web]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) [[Paper-ECCV14]](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf) [[Paper-arXiv15]](http://arxiv.org/pdf/1501.00092v1.pdf)[[Paper ICONIP-2014]](http://www.brml.org/uploads/tx_sibibtex/281.pdf)
    * Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, Learning a Deep Convolutional Network for Image Super-Resolution, in ECCV 2014
    * Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks, arXiv:1501.00092 (2015)
    * Osendorfer, Christian, Hubert Soyer, and Patrick van der Smagt. Image Super-Resolution with Fast Approximate Convolutional Sparse Coding. Neural Information Processing. Springer International Publishing, 2014. 
 * Compression Artifacts Reduction [[Paper-arXiv15]](http://arxiv.org/pdf/1504.06993v1)
   * Chao Dong, Yubin Deng, Chen Change Loy, Xiaoou Tang, Compression Artifacts Reduction by a Deep Convolutional Network, arXiv:1504.06993
 * Non-Uniform Motion Blur Removal [[Paper]](http://arxiv.org/pdf/1503.00593v3)
  * Jian Sun, Wenfei Cao, Zongben Xu, Jean Ponce, Learning a Convolutional Neural Network for Non-uniform Motion Blur Removal, CVPR 2015. 
 * Image Deconvolution [[Web]](http://lxu.me/projects/dcnn/) [[Paper]](http://lxu.me/mypapers/dcnn_nips14.pdf)
  *  Li Xu, Jimmy SJ. Ren, Ce Liu, Jiaya Jia, "Deep Convolutional Neural Network for Image Deconvolution" Advances in Neural Information Processing Systems (NIPS), 2014.
 *  Deep Edge-Aware Filter [[Paper]](http://jmlr.org/proceedings/papers/v37/xub15.pdf)
  *  Li Xu, Jimmy SJ. Ren, Qiong Yan, Renjie Liao, Jiaya Jia "Deep Edge-Aware Filters" International Conference on Machine Learning (ICML), 2015.
 * Computing the Stereo Matching Cost with a Convolutional Neural Network [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zbontar_Computing_the_Stereo_2015_CVPR_paper.pdf)
  *  Jure Žbontar, Yann LeCun, Computing the Stereo Matching Cost with a Convolutional Neural Network, CVPR 2015.

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

### Visual Attention and Saliency
  * Mr-CNN [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Predicting_Eye_Fixations_2015_CVPR_paper.pdf)
   * Nian Liu, Junwei Han, Dingwen Zhang, Shifeng Wen, Tianming Liu, Predicting Eye Fixations using Convolutional Neural Networks, CVPR, 2015.
  * Learning a Sequential Search for Landmarks [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Singh_Learning_a_Sequential_2015_CVPR_paper.pdf)
   * Saurabh Singh, Derek Hoiem, David Forsyth, Learning a Sequential Search for Landmarks, CVPR, 2015.
  * Multiple Object Recognition with Visual Attention [[Paper]](http://arxiv.org/pdf/1412.7755v2.pdf)
   * Jimmy Lei Ba, Volodymyr Mnih, Koray Kavukcuoglu, Multiple Object Recognition with Visual Attention, ICLR, 2015.
  * Recurrent Models of Visual Attention[[Paper]](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)
   * Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu, Recurrent Models of Visual Attention, NIPS, 2014.
   
### Object Recognition
  * Weakly-supervised learning with convolutional neural networks [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Oquab_Is_Object_Localization_2015_CVPR_paper.pdf)
   * Maxime Oquab, Leon Bottou, Ivan Laptev, Josef Sivic, Is object localization for free? – Weakly-supervised learning with convolutional neural networks, CVPR, 2015.
  * FV-CNN [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Cimpoi_Deep_Filter_Banks_2015_CVPR_paper.pdf)
   * Mircea Cimpoi, Subhransu Maji, Andrea Vedaldi, Deep Filter Banks for Texture Recognition and Segmentation, CVPR, 2015.
   
### Understanding CNN
  * Equivariance and Equivalence of Representations [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Lenc_Understanding_Image_Representations_2015_CVPR_paper.pdf)
   * Karel Lenc, Andrea Vedaldi, Understanding image representations by measuring their equivariance and equivalence, CVPR, 2015.
  * Deep Neural Networks Are Easily Fooled [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)
   * Anh Nguyen, Jason Yosinski, Jeff Clune, Deep Neural Networks are Easily Fooled:High Confidence Predictions for Unrecognizable Images, CVPR, 2015.
  * Understanding Deep Image Representations by Inverting Them [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.pdf)
   * Aravindh Mahendran, Andrea Vedaldi, Understanding Deep Image Representations by Inverting Them, CVPR, 2015.
   
### Other Topics
  * Surface Normal Estimation [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Designing_Deep_Networks_2015_CVPR_paper.pdf)
   * Xiaolong Wang, David F. Fouhey, Abhinav Gupta, Designing Deep Networks for Surface Normal Estimation, CVPR, 2015.
  * Action Detection [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Gkioxari_Finding_Action_Tubes_2015_CVPR_paper.pdf)
   * Georgia Gkioxari, Jitendra Malik, Finding Action Tubes, CVPR, 2015.
  * Crowd Counting [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Cross-Scene_Crowd_Counting_2015_CVPR_paper.pdf)
   * Cong Zhang, Hongsheng Li, Xiaogang Wang, Xiaokang Yang, Cross-scene Crowd Counting via Deep Convolutional Neural Networks, CVPR, 2015.
  * 3D Shape Retrieval [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Sketch-Based_3D_Shape_2015_CVPR_paper.pdf)
   * Fang Wang, Le Kang, Yi Li, Sketch-based 3D Shape Retrieval using Convolutional Neural Networks, CVPR, 2015.
  * Generate image [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf)
   * Alexey Dosovitskiy, Jost Tobias Springenberg, Thomas Brox, Learning to Generate Chairs with Convolutional Neural Networks, CVPR, 2015.  

## Courses
 * Deep Vision
  * [Stanford] [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
  * [CUHK] [ELEG 5040: Advanced Topics in Signal Processing(Introduction to Deep Learning)](https://piazza.com/cuhk.edu.hk/spring2015/eleg5040/home)
 * More Deep Learning
  * [Stanford] [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
  * [Oxford] [Deep Learning by Prof. Nando de Freitas](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
  * [NYU] [Deep Learning by Prof. Yann LeCun](http://cilvr.cs.nyu.edu/doku.php?id=courses:deeplearning2014:start)

## Books
 * Free Online Books
  * [Deep Learning by Yoshua Bengio, Ian Goodfellow and Aaron Courville](http://www.iro.umontreal.ca/~bengioy/dlbook/)
  * [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
  * [Deep Learning Tutorial by LISA lab, University of Montreal](http://deeplearning.net/tutorial/deeplearning.pdf)

## Videos
 * [Deep Learning, Self-Taught Learning and Unsupervised Feature Learning By Andrew Ng](https://www.youtube.com/watch?v=n1ViNeWhC24)
 * [Recent Developments in Deep Learning By Geoff Hinton](https://www.youtube.com/watch?v=sc-KbuZqGkI)
 * [The Unreasonable Effectiveness of Deep Learning by Yann LeCun](https://www.youtube.com/watch?v=sc-KbuZqGkI)
 * [Deep Learning of Representations by Yoshua bengio](https://www.youtube.com/watch?v=4xsVFLnHC_0)

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
  * Source code for the paper "Rich feature hierarchies for accurate object detection and semantic segmentation", CVPR 2014. [[Web]](https://github.com/rbgirshick/rcnn)
  * Source code for the paper "Fully Convolutional Networks for Semantic Segmentation", CVPR 2015. [[Web]](https://github.com/longjon/caffe/tree/future)
 * Super-Resolution
  * Image Super-Resolution for Anime-Style-Art [[Web]](https://github.com/nagadomi/waifu2x)
 * Edge Detection
  * Source code for the paper "DeepContour: A Deep Convolutional Feature Learned by Positive-Sharing Loss for Contour Detection" CVPR 2015. [[Web]](https://github.com/shenwei1231/DeepContour)
 
## Tutorials
  * [CVPR 2014] [Tutorial on Deep Learning in Computer Vision](https://sites.google.com/site/deeplearningcvpr2014/)
  * [CVPR 2015] [Applied Deep Learning for Computer Vision with Torch](http://torch.ch/docs/cvpr15.html)

## Blogs
* [Deep down the rabbit hole: CVPR 2015 and beyond@Tombone's Computer Vision Blog](http://www.computervisionblog.com/2015/06/deep-down-rabbit-hole-cvpr-2015-and.html)
* [CVPR recap and where we're going@Zoya Bylinskii (MIT PhD Student)'s Blog](http://zoyathinks.blogspot.kr/2015/06/cvpr-recap-and-where-were-going.html)
* [Facebook's AI Painting@Wired](http://www.wired.com/2015/06/facebook-googles-fake-brains-spawn-new-visual-reality/)
* [Inceptionism: Going Deeper into Neural Networks@Google Research](http://googleresearch.blogspot.kr/2015/06/inceptionism-going-deeper-into-neural.html)
