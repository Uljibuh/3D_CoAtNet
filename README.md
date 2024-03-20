# 3D_CoAtNet

orginal paper of CoAtNet "https://arxiv.org/pdf/2106.04803.pdf"


the 3D relative position bias is adopted from Video swin transformer
github link: https://github.com/SwinTransformer/Video-Swin-Transformer

the model architecture is mostly inspired by Implmentation of 2D CoAtNet by https://github.com/KKKSQJ

github link: https://github.com/KKKSQJ/DeepLearning/blob/master/classification/coatNet/models/networks.py

I converted the 2D CoAtNet in to 3D version,and i tested the model on 3D medical image classificaiton task.
It works even on small dataset, around 400 3D images on training set. and result is compareble with 3D EfficientNet
