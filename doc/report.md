---
# Compile with `pandoc report.md -o report.pdf --highlight-style zenburn`
title: Neural Netoworks Project 3 - Multilayer Perceptron with backpropagation
geometry: margin=4cm
toc: true
author:
  - Antoni Szczepanik
  - Ajewole Adedamola Jude

date: 6th June 2021, Warsaw
include-before: |
	\newpage
---


\newpage

# Problem description

Even though computers can replace a lot of human labor there are still some 
domains which are very hard to automate. Up until recently, the hard to automate
domains included voice and image recognition, classification tasks and a few
others. Today, this problems are solved successfully with machine learning,
in particular with the use of neural networks.

Neural networks, are algorithms inspired by the biological neural networks 
that are present in human brains. Such network is a collection of connected
units called artificial neurons, which loosely mimic neurons in biological 
brain. Each connection, like the synapses in a biological brain, is responsible
for transmitting a signal to other neurons.

Multilayer perceptron is possibly one of the simples neural networks created.
Even though it is very simple in principle, the inner workings of it are
quite complex and interesting. 

In this task we are asked to create a simple MLP from scratch, 
using only libriaries that allow primitive matrix operations. Most importantly
we are asked to implement backpropagation algorithm which will allow the
network to adjust it's parameter to fit the task.

\newpage

# How to run the application?
To run the project with exactly the same package versions as we did one can
use any Python virtual env manager.  All requirements are frozen in `requirements.txt` file.
For example using Python built-in venv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
After that one can start `jupyterlab` with:

```bash
python -m jupyterlab
```

All experiments are present in `experiment.ipynb` notebook.
The neural network source is present in `toygrad.py`.
Additional plotting helper funcitons are placed in `plot.py`.

\newpage

# Theoretical introduction

Even though MLP is very simple in principle, the inner workings of it are
quite complex and interesting. Especially the training process is nontrivial.

Every MLP is organised into layers of neurons.
Neurons in each layer are connected to all neurons in the previous layer and
all neurons in the following layer.

![Sample multilayer perceptron scheme](mlp_wiki.png){ width=250px }

The first layer is called input layer. This layer will take the input we will
provide it with. For example in case of image recognition it would be a vector
representing the image. After that its neurons will fire passing signals to the next
layer. The same will happen for the following layers, until the signal is passed
to the last layer the "output". This is our resulting signal. Depending
on the task they may be one or many output neurons. Interpretation of network
result will also be dependent on the task - the results may be probabilities,
or real values, with any number of output signals.

At each neuron it's input signals are multiplied by given connections
weight, the bias is added and later the activation function is applied.

Initially weights and biases are initalized randomly from $(0, 1)$ uniform
distribution. This however makes our network draw random conclusions as well.
We need a way to indroduce updates to weights and biases to allow our network
to "learn".
Hence, we could treat neural network as an optimization problem. The weights
and biases are the thing that could be adjusted, and we optimize for
getting the lowest error possible on the network output.

## Training algorithm

The broad overview of the training algorithm is as follows:

1. Pass a single input through the network. (forward pass)

2. Measure error at the output layer using loss function.

3. Calculate updates for weights and biases based on the error (using 
backpropagation) and update the weights. (backward pass)

4. Repeat the procedure for all training samples.

5. Repeat the procedure based on the whole dataset multiple times called epochs.

### Backpropagation

The algorithm that is used for adjusting the weights of the network is called
backpropagation. It allows to slighly adjust all of the weights (and the
biases, if present) based on error between the output layer and actual values.
It computes the gradient of the loss function with respect to the weights of the
network for a single input–output example. 
In principle the backpropagation algorithm works by computing the gradient of
the loss function with respect to each weight by the chain rule, computing the gradient
one layer at a time, iterating backward from the last layer to avoid redundant calculations
of intermediate terms in the chain rule.
For a more detailed description please refer to TODO:[1]

The algorith is parallelizable and efficient, which makes it feasible to use
for training multilayer networks, updating weights to minimize loss.

\newpage

# Experiments

## Impact of various activation functions on accuracy

To test impact of activation functions on accuracy we took a look at 3 activation functions and 2 datasets.
The specified activation function was chosen in every layer except the last one.

1. Sigmoid
2. ReLU
3. TanH

All test where performed on dataset of size 1000 to speed up the computations. If the time allowed
to calculate the results on datasets of bigger sizes it would be suggested to do that in the 
future.

###  Simple dataset - binary classification

Apart from the activation functions all other parameters are fixed as follows:

```bash
layers = [
        Layer(in_size=2, out_size=5, activ_function=activation_function),
        Layer(in_size=5, out_size=5, activ_function=activation_function),
        Layer(in_size=5, out_size=1, activ_function=Sigmoid),
    ]
    m = MLP(
        layers=layers,
        loss=BinaryCrossEntropy,
        bias=True,
        batch_size=16,
        epochs=100,
        momentum=0.001,
        learning_rate=0.5,
    )

```

`activation_function` is just one of the three specified activations.

The losses on corresponding test dataset are as follows:

```bash
=======================================================================================
Simple dataset (train/test sample size 1000)
=======================================================================================
Sigmoid  - train loss 0.032 (std:0.02) - test loss 0.033 (std:0.11)
ReLU     - train loss 0.025 (std:0.04) - test loss 0.031 (std:0.21)
TanH     - train loss 0.020 (std:0.04) - test loss 0.011 (std:0.08)
```

We can clearly see that the best performance is achieved by TanH activation in
this case. This is also confirmed by low loss standard deviation on the test set.

If we inspect the decision boundary of the network our suspicions are further confirmed.

![Decision boundary of Sigmoid activations - Simple dataset](decisions_sig_simple.png){ width=250px }
![Decision boundary of ReLU activations - Simple dataset](decisions_rel_simple.png){ width=250px }
![Decision boundary of TanH activations - Simple dataset](decisions_tanh_simple.png){ width=250px }

###  Cube dataset - regression

When performing the regression test on Cube dataset we also fixed all 
other network parameters. They present in the following way.

```bash
layers = [
	Layer(in_size=1, out_size=8, activ_function=activation_function),
	Layer(in_size=8, out_size=8, activ_function=activation_function),
	Layer(in_size=8, out_size=1, activ_function=Linear),
]
m = MLP(
	layers=layers,
	loss=AbsoluteError,
	bias=True,
	batch_size=1,
	epochs=100,
	momentum=0.99,
	learning_rate=0.01,
)

```

`activation_function` is just one of the three specified activations.

The losses on corresponding test dataset are as follows:

```bash
=======================================================================================
Simple dataset (train/test sample size 1000)
=======================================================================================
Sigmoid  - train loss 0.054 (std:0.04) - test loss 0.044 (std:0.04)
ReLU     - train loss 0.163 (std:0.14) - test loss 0.269 (std:0.11)
TanH     - train loss 0.089 (std:0.07) - test loss 0.060 (std:0.03)
```

We can clearly see that the best performance is achieved by Sigmoid activation in
this case. We may say that the Sigmoid activation function works better in the
case of regression in case of this particular dataset.

This confirms that there are no clear rule about selection of activation functions.

We could also inspect predicions on the test dataset for every activation.

![Results of Sigmoid activations - Cube dataset](decisions_sig_cube.png){ width=250px }
![Results of ReLU activations - Cube dataset](decisions_rel_cube.png){ width=250px }
![Results of TanH activations - Cube dataset](decisions_tanh_cube.png){ width=250px }


## Impact of number of hidden layers and their size on accuracy

To analyze the impact of number and size of hidden layers all remaining parematers
were fixed. Apart from that the grid of possible architectures was tested with
discrete hidden layer number and hidden layer size paramters.

```bash
hidden_layers = [1, 3, 5]
hidden_layer_sizes = [3, 5, 10]
```

### Three Gauss dataset

The following parameters were fixed when performing layer experiments on Gauss dataset:

```bash
 layers = [
            Layer(in_size=2, out_size=hidden_layer_size, activ_function=Sigmoid),
	    ... hidden layers of appropriate sizes ...
            Layer(in_size=hidden_layer_size, out_size=3, activ_function=SoftMax),
        ]
        m = MLP(
            layers=layers,
            loss=CategoricalCrossEntropy,
            bias=True,
            batch_size=16,
            epochs=100,
            momentum=0.1,
            learning_rate=0.2,
        )

```

The results are presented in the table below. Column names correspond to number of hidden layers.
Row names correspond to number of neurons in each hidden layer.


```bash
 		1 			3 			5
3 	0.178 (std: 0.40) 	0.367 (std: 0.02) 	0.367 (std: 0.02)
5 	0.078 (std: 0.18) 	0.367 (std: 0.02) 	0.367 (std: 0.03)
10 	0.078 (std: 0.20) 	0.367 (std: 0.02) 	0.370 (std: 0.05)
```

We can see that the network learns best in case of having only a single hidden layer.
Especially the single hidden layer size 5 and 10 produce promising results.

It is possible that if we adjusted other network parameters we would get 
better result for larger networks, as they didn't have a chance to learn.
(the learning rate or the momentum were to small).

### Activation dataset

The following parameters were fixed when performing layer experiments on Activation dataset:

```bash
layers = [
            Layer(in_size=1, out_size=hidden_layer_size, activ_function=Sigmoid),
	    ... hidden layers of appropriate sizes ...
            Layer(in_size=hidden_layer_size, out_size=1, activ_function=Linear),
        ]
        m = MLP(
            layers=layers,
            loss=AbsoluteError,
            bias=True,
            batch_size=16,
            epochs=70,
            momentum=0.1,
            learning_rate=0.01,
        )

```

The results are presented in the table below. Column names correspond to number of hidden layers.
Row names correspond to number of neurons in each hidden layer.


```bash
 		1 			3 			5
3 	4.596 (std: 8.19) 	109.870 (std: 57.15) 	109.703 (std: 59.94)
5 	27.830 (std: 21.04) 	109.718 (std: 59.91) 	109.743 (std: 59.07)
10 	109.669 (std: 63.01) 	109.679 (std: 60.51) 	109.683 (std: 60.37)
```

Once again the best result are achieved with small network and hidden layer size.

# Conclusions

We successfuly experimented with both network architecture as well as activation functions.
Unfortunatelly it is very hard to propose specific conclusions. Neural network hyperparameters
seem to be very dependent on specific datasets, and therefore another parameters may
work better in case of different cases.

# References
1. Andrej Karpathy Micrograd implementation.
   Retrieved from: https://github.com/karpathy/micrograd
2. MLPs with Backpropagation. CS 472 – Backpropagation. 
   Retrieved from https://axon.cs.byu.edu/~martinez/classes/478/slides/BP.pdf
3. Brownlee, J. (2016). How to Code a Neural Network with Backpropagation In Python (from scratch). 
   Machine Learning Mastery.
   Retrieved from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
