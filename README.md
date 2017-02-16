# Fork of **tiny-dnn**

##  Please follow the link below for the upstream project if you are looking for tiny-dnn

http://github.com/tiny-dnn/tiny-dnn/

If you like C++ and wants to play with deep learning, in my opining, this library is good with the following nice
features:

* No libraries or external packages required so I could had it up and running in a few minutes.
* Classical examples included (MNIST and CIFAR10), so it's easy to get started.
* Easy to understand interface.
* The project is active with good support as of this writing Feb 2017
* Many others... follow the upstream project.

## Immediate goal of this fork: improve MNIST single-digit classification performance in the upstream code

The advertized performance by the upstream is 98.8%. 
The current performace in this fork is 99.1% 99.2% is ready to be commited.

The figure below is an example of a plot for simulation results. It shows the number of correct classifications out of
10000 tests as the network gets trained.

<img src="https://cloud.githubusercontent.com/assets/23116478/22905779/7b05721e-f1f6-11e6-83a2-a7474d7a1d41.png">
(Number of correct digit classifications versus epoch)

The core of the codebase is to be preserved such that documentation related to on how to compile, etc., is available
in the upstream. 

<img src="https://travis-ci.org/pliptor/tiny-dnn.svg?branch=master">

## References
[1] Y. Bengio, [Practical Recommendations for Gradient-Based Training of Deep Architectures.](http://arxiv.org/pdf/1206.5533v2.pdf) 
    arXiv:1206.5533v2, 2012

[2] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, [Gradient-based learning applied to document recognition.](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
    Proceedings of the IEEE, 86, 2278-2324.
    
other useful reference lists:
- [UFLDL Recommended Readings](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Recommended_Readings)
- [deeplearning.net reading list](http://deeplearning.net/reading-list/)

## License
The BSD 3-Clause License (keeping upstream license)

