# Notes(Playing in torch)

## History

* On a conceptual level, deep learning is inspired by the brain but not all of the brain’s details are relevant. For a comparison, aeroplanes were inspired by birds. The principle of flying is the same but the details are extremely different.
* Deep Learning took off again in 1985 with the emergence of backpropagation. In 1995, the field died again and the machine learning community abandoned the idea of neural nets. In early 2010, people start using neuron nets in speech recognition with huge performance improvement and later it became widely deployed in the commercial field. In 2013, computer vision started to switch to neuron nets. In 2016, the same transition occurred in natural language processing. Soon, similar revolutions will occur in robotics, control, and many other fields
* In Old times,the standard model of pattern recognition consists of feature extractor and trainable classifier. Here, a trainable classifier could be a perceptron or single neural network. The problem is feature extractor should be engineered by hand. Which means, pattern recognition/computer vision focus on feature extractor considering how to design it for a particular problem, not much devoted to a trainable classifier
* After the emergence and development of deep learning, the 2-stage process changed to the sequences of modules. Each module has tunable parameters and nonlinearity. Then, stack them making multiple layers. This is why it is called “deep learning”. The reason why using nonlinearity rather than linearity is that two linear layers could be one linear layer since the composition of two linear is linear.

## CNN Evolution

* In animal brains, neurons react to edges that are at particular orientations. Groups of neurons that react to the same orientations are replicated over all of the visual field.
* Fukushima (1982) built a neural net (NN) that worked the same way as the brain, based on two concepts. First, neurons are replicated across the visual field. Second, there are complex cells that pool the information from simple cells (orientation-selective units). As a result, the shift of the picture will change the activation of simple cells, but will not influence the integrated activation of the complex cell (convolutional pooling)
* LeCun (1990) used backprop to train a CNN to recognize handwritten digits. There is a demo from 1992 where the algorithm recognizes the digits of any style. Doing character/pattern recognition using a model that is trained end-to-end was new at that time. Previously, people had used feature extractors with a supervised model on top.
* Still: Why do architectures with multiple layers perform better, given that we can approximate any function with two layers? Why do CNNs work well with natural data such as speech, images, and text? How are we able to optimize non-convex functions so well? Why do over-parametrised architectures work?

## Features

* It seems like we can approximate any function with 2 layer network..Even then we need deep Learning!!!

  Because : The problem with 2 layer fallacy is that the complexity and size of the middle layer is exponential in N.(to do well with a difficult task, need LOTS of templates). But if you expand the number of layers to log(*N*), the layers become linear in *N*. There is a trade-off between time and space.

* Hierarchy in representation



## LA

* 

## Classification Basic

* 

## Regression Basic

* 

## Convnets:

* In detection many methods doesn't be overly concerned with the precise location of the object in the image.

* CNNs systematize this idea of spatial invariance, exploiting it to learn useful representations with fewer parameters.

* We need network to use Stationarity(translation invariance),Compositionality and locality properties....Which together know as  properties of natural properties.

* Let us invoke Locality first:

  * Locality => Sparsity
    * If our data exhibits locality, each neuron needs to be connected to only a few local neurons of the previous layer. Thus number of connections reduces...
  * Stationarity => parameter Sharing
    *  Because of this property we use same a small set of parameters multiple times across the network architecture.
    *   Advantages of Parameter Sharing and Sparsity:
      * Parameter Sharing:
        * Faster Convergence
        * Better Generalisation
        * not constrained to input size
        * kernel independence ⇒> high parallelisation
      * Sparsity:
        * reduced amount of computation

* Feature map: the convolutional layer output is sometimes called a feature map.

* Receptive field: receptive field refers to all the elements (from all the previous layers) that may affect the calculation of  x  during the forward propagation.

* For More Details about operations in CNN's refer [Convnet_basic.ipynb](https://github.com/Jayanth49/Introductory_excersices_ml/blob/master/Deep_Learning/Convnets/Convnets_basic.ipynb)

* For Implementation details refer [convnet_intro](https://github.com/Jayanth49/Introductory_excersices_ml/blob/master/Deep_Learning/Convnets/convnet_intro.ipynb).

* For experiments refer [modified_first_cnn](https://github.com/Jayanth49/Introductory_excersices_ml/blob/master/Deep_Learning/Convnets/modified_first_cnn.ipynb)

  

### Modern CNNS

### LeNet 

````mermaid
graph TB
	Image --> B(Convolution Layer-1)
	B --> C(Convolution Layer-2)
	C --> D(Fully connected Layer-1)
	D --> E(fully connected Layer-2)
	E --> F(Softmax)
````

* Initially we had 28*28 image

* **Convolution Layer-1:**

  * Kernel Size: 5*5

  * Number of kernels = 6
  * stride = 1
  * padding = 0
  * output feature-map size = 6@24*24
  * ReLU Activation
  * Average_Pooling
  * Final feature-map size = 6@12*12

* **Convolution Layer-2:**

  * Kernel Size : 5*5
  * Number of kernels = 6
  * stride = 2
  * padding = 0
  * output feature-map size = 6@8*8
  * ReLU Activation
  * Average_Pooling
  * Final feature-map size = 6@4*4

* **Fully Connected  Layer-1:**

  * 120 neurons connected with previous layer

* **Fully Connected Layer-2:**

  * 84 neurons connected with previous layer

* Finally doing **Softmax**.

### AlexNet 

* 



###  ZFNet

###   

### GoogLeNet 



* VGGNet 
* ResNet 
* Nin 
* Densenet 
* Inception Net.
* Xception Net.









