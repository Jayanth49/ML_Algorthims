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

## Batch Normalization

* For Fully Connected-Layers
  * When applying batch normalization to fully-connected layers, the original paper inserts batch normalization after the affine transformation and before the nonlinear activation function.
  * (later applications may insert batch normalization right after activation functions)
  * 



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

* Although LeNet achieved good results on early small datasets, the performance and feasibility of training CNNs on larger, more realistic datasets had yet to be established.
* Traditional methods (using support vectors):
  * Obtain an interesting dataset. In early days, these datasets required expensive sensors (at the time, 1 megapixel images were state-of-the-art)
  * Preprocess the dataset with hand-crafted features based on some knowledge of optics, geometry, other analytic tools, and occasionally on the serendipitous discoveries of lucky graduate students.
  * Feed the data through a standard set of feature extractors such as the SIFT (scale-invariant feature transform) [[Lowe, 2004\]](https://d2l.ai/chapter_references/zreferences.html#lowe-2004), the SURF (speeded up robust features) [[Bay et al., 2006\]](https://d2l.ai/chapter_references/zreferences.html#bay-tuytelaars-van-gool-2006),  HOG (histograms of oriented gradient) [[Dalal & Triggs, 2005\]](https://d2l.ai/chapter_references/zreferences.html#dalal-triggs-2005), [bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision) or any number of other hand-tuned pipelines.
  * Dump the resulting representations into your favorite classifier, likely a linear model or kernel method, to train a classifier



### VGG net

* Alexnet doesn't provide  a general template to guide subsequent researchers in designing new networks.
* One VGG block consists of a sequence of convolutional layers, followed by a max pooling layer for spatial downsampling

````python
	import torch
	from torch import nn
    
    def vgg_block(num_convs, in_channels, out_channels):
    layers=[]
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
````

* VGG-11 constructs a network using reusable convolutional blocks. Different VGG models can be defined by the differences in the number of convolutional layers and output channels in each block.
* The use of blocks leads to very compact representations of the network definition. It allows for efficient design of complex networks.
* **In particular, here the creators found that several layers of deep and narrow convolutions (i.e., 3*3) were more effective than fewer layers of wider convolutions.**

###  NiN

* Previous 3 are almost similar design/pattern like extract features exploiting *spatial* structure via a sequence of convolution and pooling layers and then post-process the representations via fully-connected layers.

* NiN uses convolutional layers with window shapes of 11×11,5×5,3×3, and the corresponding numbers of output channels are the same as in AlexNet. Each NiN block is followed by a maximum pooling layer with a stride of 2 and a window shape of 3×3.

* One significant difference between NiN and AlexNet is that NiN avoids fully-connected layers altogether

*  Instead, NiN uses an NiN block with a number of output channels equal to the number of label classes, followed by a *global* average pooling layer, yielding a vector of logits.

* One advantage of NiN’s design is that it significantly reduces the number of required model parameters. However, in practice, this design sometimes requires increased model training time.

  * **Idea of NiN**:

    * The convolution filter in CNN is a generalized linear model (GLM) for the underlying data patch, and we argue that the level of abstraction is low with GLM.(unclear statement)
    * In NIN, the GLM is replaced with a ”micro network” structure which is a general nonlinear function approximator.
    * maxout network??
    * **mlpconv:**
      * In normal Linear convolution we do convolution and add them up to get a pixel in output, but here we keep a mlp in place of convolution layer.

    * **Global Averaging Pooling:**
      * In traditional CNN architectures, the feature maps of the last convolution layer are flattened and passed on to one or more fully connected layers, which are then passed on to softmax logistics layer for spitting out class probabilities
      * and we know that number of parameters here are huge ,and high possibiltiy of overfitting.
      * the last MLPconv layer produces as many activation maps as the number of classes being predicted. Then, each map is averaged giving rise to the raw scores of the classes. These are then fed to a SoftMax layer to produce the probabilities, totally making FC layers redundant
      * The mapping between the extracted features and the class scores is more intuitive and direct. The feature can be treated as category confidence.
      * An implicit advantage is that there are no new parameters to train (unlike the FC layers), leading to less overfitting.

* [implementation_helper](https://stats.stackexchange.com/questions/273486/network-in-network-in-keras-implementation)

  NiN block:

  ````python
  import torch
  from torch import nn
  
  def nin_block(in_channels, out_channels, kernel_size, strides, padding):
      return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
          nn.ReLU(),
          nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
          nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
  ````

### GoogLeNet 

* The basic convolutional block in GoogLeNet is called an *Inception block*

*  The inception block consists of four parallel paths.

*  Finally, the outputs along each path are concatenated along the channel dimension and comprise the block’s output

*  The commonly-tuned hyperparameters of the Inception block are the number of output channels per layer.

* **Inception Block**:

  ````python
  import torch
  from torch import nn
  from torch.nn import functional as F
  
  class Inception(nn.Module):
      # `c1`--`c4` are the number of output channels for each path
      def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
          super(Inception, self).__init__(**kwargs)
          # Path 1 is a single 1 x 1 convolutional layer
          self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
          # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
          # convolutional layer
          self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
          self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
          # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
          # convolutional layer
          self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
          self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
          # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
          # convolutional layer
          self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
          self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
  
      def forward(self, x):
          p1 = F.relu(self.p1_1(x))
          p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
          p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
          p4 = F.relu(self.p4_2(self.p4_1(x)))
          # Concatenate the outputs on the channel dimension
          return torch.cat((p1, p2, p3, p4), dim=1)
  ````

  

* GoogLeNet uses a stack of a total of 9 inception blocks and global average pooling to generate its estimates. Maximum pooling between inception blocks reduces the dimensionality.

* The Inception block is equivalent to a subnetwork with four paths. It extracts information in parallel through convolutional layers of different window shapes and maximum pooling layers. 1×11×1 convolutions reduce channel dimensionality on a per-pixel level. Maximum pooling reduces the resolution.

* GoogLeNet connects multiple well-designed Inception blocks with other layers in series. The ratio of the number of channels assigned in the Inception block is obtained through a large number of experiments on the ImageNet dataset.

* GoogLeNet, as well as its succeeding versions, was one of the most efficient models on ImageNet, providing similar test accuracy with lower computational complexity.



### ResNet





### Densenet 
* Inception Net.
* Xception Net.









