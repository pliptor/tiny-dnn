# tiny-dnn-lab:<br> A fork of the C++ Deep Learning library <img src="https://github.com/pliptor/tiny-dnn/blob/master/docs/logo/TinyDNN-logo-letters-alpha-version.png" alt="Drawing" width="150">

## This fork plays with some of its examples and create others

<img src="https://travis-ci.org/pliptor/tiny-dnn.svg?branch=master">

###  Please follow the link below for the upstream project if you are looking for tiny-dnn

https://github.com/tiny-dnn/tiny-dnn/

If you like C++ and want to play with deep learning, in my opinion, tiny-dnn is a good library for the following nice
features:

* No libraries or external packages required. I could had it up and running in a few minutes.
* Classical examples included (MNIST and CIFAR10), so it's easy to get started.
* Easy to understand interface.
* The upstream library project is active with good support as of this writing Feb 2017
* Many others... follow the upstream project.

## Immediate goal of this fork: improve MNIST classification performance in the upstream code

### The challenge is to classify handwritten single digits like this
<img src="https://cloud.githubusercontent.com/assets/23116478/23090826/1b412d8c-f55c-11e6-899b-eea967f80709.png">

The advertized performance by the upstream is 98.8%. 
The current performace in this fork is 99.1% 99.2% is ready to be commited.

The figure below is an example of a plot for simulation results. It shows the number of correct classifications out of
10000 tests as the network gets trained.

<img src="https://cloud.githubusercontent.com/assets/23116478/22905779/7b05721e-f1f6-11e6-83a2-a7474d7a1d41.png">
*(Number of correct digit classifications versus epoch)*


### The next challenge is classifying images (cifar10)

Here's snapshot generated when the network is being used to recognize a picture of horse.
<img src="https://cloud.githubusercontent.com/assets/23116478/23090774/f33b0b88-f55a-11e6-9742-22c67b2ea7ba.png">

## References
[1] Y. Bengio, [Practical Recommendations for Gradient-Based Training of Deep Architectures.](http://arxiv.org/pdf/1206.5533v2.pdf) 
    arXiv:1206.5533v2, 2012

[2] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, [Gradient-based learning applied to document recognition.](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
    Proceedings of the IEEE, 86, 2278-2324.
    
other useful reference lists:
- [UFLDL Recommended Readings](https://deeplearning.stanford.edu/wiki/index.php/UFLDL_Recommended_Readings)
- [deeplearning.net reading list](https://deeplearning.net/reading-list/)

## License
The BSD 3-Clause License (keeping upstream license)

