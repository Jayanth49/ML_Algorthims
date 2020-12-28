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

## Convnets

* 

### Modern CNNS





