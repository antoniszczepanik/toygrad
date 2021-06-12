---
# Compile with `pandoc report.md -o report.pdf --highlight-style zenburn`
title: Appendix -  Neural Netoworks Project 3 - training MLP on MNIST dataset
geometry: margin=4cm
author:
  - Antoni Szczepanik
  - Ajewole Adedamola Jude

date: 11th June 2021, Warsaw

---

# Problem description

The task for the second part of the assignment is to train previously created
MLP on MNIST dataset. This is a dataset of annotated handwritten digits.
By inspecting a few of the entries we can get inutive feel of the dataset.

![MNIST dataset sample](mnist_sample.png){ width=400px }

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
MNIST model training and analysis is done based in `mnist.ipynb` notebook.

# Network Development

### Measuring accurracy

As defined on Kaggle platform the metric against which our solution will be
evaluated is accuracy. Accurracy is a very intuitive metrix which is defined
as ratio of number of correct predictions and number of all samples.

\begin{equation}
	accuracy = \frac{\text{Number of correct prections}}{\text{Total number of predictions}}
\end{equation}


### Train/Test split

We decide to perform a simple split of our data into a training set (80%),
and testing/validation set (20%)

All Y values have been one hot encoded to allow comparing between network outputs.

### Initial architecture

At first we decide to perform a single training with very simple, almost default
parameters. This will allow us to benchmark other hyper-parameter combinations 
against it.


```bash
model parameters and results here
```


### Adding normalisation
### Automatic Hyperparameter tuning
### Trying to gain and

# Conclusions

