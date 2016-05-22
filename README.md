# Fisher Caffe
C++ implementation of learning Fisher vectors using VlFeat. In addition the fisher vector can be concatenated with another feature vector such as a Caffe layer etc. 

## Requirements
* CMake
* Boost
* Eigen
* OpenCV
* <a href=http://www.vlfeat.org/>VlFeat</a>

## Training
* Extract Dense SIFTs from a set of labeled images
* PCA Project
* Compute GMM using EM
* Compute Fisher Vector for the images from the trained GMM
* Add additional features.
* Train one-against-many linear SVM for each class

## Test
* Extract Dense SIFTs from the set of test images
* PCA Project
* Compute GMM using EM
* Compute Fisher Vector for the images from the trained GMM
* Add additional features.
* Classify using linear SVM


This code is provided as is. If you notice any errors or bugs file a request or inbox me. 
