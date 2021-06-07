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
estimates, etc.


At each neuron it's input signals are multiplied by given connections
weight, the bias is added and later the activation function is applied.

Initially weights and biases are initalized randomly from (0, 1) uniform
distribution. This however makes our network draw random conclusions as well.
We need a way to indroduce updates to weights and biases to allow our network
to "learn".
Hence, we could treat neural network as an optimization problem. The weights
and biases are the thing that could be adjusted, and we optimize for
getting the lowest error possible on the network output.



# Experiments

## Impact of various activation functions on accuracy

TODO: how does activation function affect the model’s accuracy?
TODO: Experiment with sigmoid and twoother activation functions.  
TODO: The activation function in an output layer should be chosen accordingly to the problem

### Clasification Results
### Regression Results

## Impact of number of hidden layers and their size on accuracy

TODO: how does the number of hidden layers and number of neurons in hidden 
layers impact themodel’s accuracy?  

TODO: Analyze different architectures

### Clasification Results
### Regression Results

## Impact of various loss functions  on accuracy

### Clasification Results
### Regression Results

# Conclusions
TODO: Including reasons for success/failure, further research proposals...
