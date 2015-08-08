# easyDL
Easy and fast deep learning in MATLAB.<br />
Currently, easyDL supports supervised deep neural networks only.
You can easily configure parameters of all layers with a model signature.
Please refer to below examples.<br />
Copyright (c) 2015 Taehoon Lee

# Usage
easyDL works in two different modes.

### training
`model = EASYDL(data, labels, model or model signature, options)`<br />
trains a supervised deep neural network with data-label pairs and returns the model.<br />
`model = EASYDL(data, labels, model or model signature, options, testdata, testlabels)`<br />
works in the same manner except that the testing is performed after each epoch.

### testing
`output = EASYDL(model, testdata, n)`<br />
returns feed-forward values of testdata on the n-th layer in the model.<br />
`output = EASYDL(model, testdata)`<br />
if the n is omitted, output is the last layer's activations.

### model signatures
There are three types of layers: convolutional(`C`), pooling(`P`), and fully-connected(`F`) type.
You can designate type of individual layers with a cell type variable called a *model signature*.
For example,
<li> `{'F:100', 'F'}` denotes a hidden layer with 100 units followed by a softmax output layer.
The number of units must be provided in all F layers except the softmax layer.
In the last layer, the number of units is automatically set to the number of classes. </li>
<li> `{'C:10@5x5', 'P:2x2', 'F'}` means a convolutional layer having 10 feature maps of size 5x5,
a pooling layer with 2x2 mask, and a softmax layer. </li>

### default options
<li> `alpha` (learning rate): an initial value is 0.1 and it is annealed by factor of two after 10 epochs. </li>
<li> `momentum`: an initial value is 0.5 and it is changeed to 0.95 after 20 iterations. </li>
<li> `minibatch`: 100 </li>
<li> `weightDecay`: 1e-4 </li>

# MNIST Examples

### dataset preparation
The MNIST dataset can be found [here](http://yann.lecun.com/exdb/mnist/).<br />
There are two matrices and two vectors: 
<li> images : (28 x 28 x 1 x 60000) matrix. </li>
<li> labels : (60000 x 1) vector which ranges from 1 to 10. </li>
<li> testImages : (28 x 28 x 1 x 10000) matrix. </li>
<li> testLabels : (10000 x 1) vector. </li>

### example 1: two fully connected hidden layers + a softmax output.
The numbers of nodes here are 784, 200, 100, and 10.
```
clear('options');
% set init learning rate to 0.1 and
% anneal it by factor of two after 3 epochs
options.alpha = '0.1, 0.5@3';
options.epochs = 15;
fcn = easyDL(images, labels, {'F:200', 'F:100', 'F'}, options);
pred = easyDL(fcn, testImages);
disp(sum(testLabels==pred) / length(pred));
```
This configuration and options gives 98.07% accuracy.
And the elapsed time is 51 seconds in my PC (i5-3570K, 3.4GHz).<br />

### example 2: a convolutional layers + a pooling layer + a softmax output.
```
clear('options');
options.alpha = 0.1;
options.epochs = 3;
options.weightdecay = 1e-5;
cnn = easyDL(images, labels, {'C:12@9x9', 'P:2x2,max', 'F'}, options);
pred = easyDL(cnn, testImages);
disp(sum(testLabels==pred) / length(pred));
```
The example 2 produces 98.29% accuracy and runs in 3 minutes.

### example 3: two convolutional and two pooling layers + a hidden layer + a softmax output.
```
clear('options');
% set init learning rate to 0.1 and
% anneal it by factor of two after 4 epochs
options.alpha = '0.1, 0.5@4';
options.epochs = 20;
cnn = easyDL(images, labels, {'C:8@5x5', 'P:2x2,max', 'C:12@5x5', 'P:2x2,max', 'F:100', 'F'}, options, testImages, testLabels);
pred = easyDL(cnn, testImages);
disp(sum(testLabels==pred) / length(pred));
```
The example 3 produces 99.04% accuracy and runs in 1 hour.