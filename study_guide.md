# Machine Learning Exam 3 Study Guide

## Logistic Regression

### Classification
- Discrete and supervised
- Good for when dependent variable is binary (yes/no, red/blue, etc)
- Determines probability of the binary classification (y val of sigma curve)
  ![](resources/logistic/logistic_func.png)

### *Loss Function
- "Cost function"
![](resources/logistic/image18.png)
![](resources/logistic/logistic_cost_func.png)
![](resources/logistic/logistic_cost_g2.png)
![](resources/logistic/logistic_cost_g1.png)
![](resources/logistic/image12.png)

### Compare with Linear Regression
- logistic regression predicts whether something is true or false, while linear regression predicts something of continuous size
- Linear fits a straight line, while logistic fits an "S" shaped logistic curve
- Logistic curve shows likelihood of discrete classifications

---
## Multi-Layered Neural Networks

- hidden layers are <i>neuron nodes stacked in between inputs and outputs</i>, allowing neural networks to learn more complicated features (such as <i>XOR</i> logic)

### Loss functions
Squared Error: ![](resources/multilayered_nn/image6.png)

### Tanh Activation Function and Motivation
```
h_i <- tanh(w_i + x_hat)
```

### * Forward Propagation to compute output
- Compute activation of each hidden node by taking the tanh of the weighted sum of inputs (``` summation( w_i * x ```))
- Output = sum of hidden node activations * weights

#### Steps w/ Example:
1. Compute activation of each hidden node:
    ![](resources/multilayered_nn/image7.png)
2. Compute output value:
   ![](resources/multilayered_nn/image21.png)
3. Find error (```expected - actual```)
    ![](resources/multilayered_nn/image3.png)
    > In this example, the expected value was 0 (see table in first step). 0 - 0.039 = -0.039

### Backward Propagation to Compute Weights
- Backpropagation is a procedure to repeatedly adjust the weights of a multilayer perceptron to minimize the difference between actual output and desired output
- ```backpropagation = gradient descent + chain rule```
- ``` e_n(error on the nth example) = y_n - summation(h_i * x_i)```

#### Steps w/ Example
1. Compute gradient change for weights from hidden to output nodes (```g = eh```) and new heights
    ![](resources/multilayered_nn/image26.png)
    ![](resources/multilayered_nn/image22.png)
    > above equation = 0.817
2. Compute new weights from input to hidden 
    ![](resources/multilayered_nn/image25.png)
    ![](resources/multilayered_nn/image15.png)


### Inductive Bias
![](resources/multilayered_nn/image14.png)
- smooth interpolation
  - any two data points with same class values, any other points between them will have same class value
- Large weight values can make the network over adjust to minor differences.
  - If some weights drom to 0, they can drop out and won't affect the function. This can be good to avoid overfitting

### * Hyperparameters and Impact on Underfitting/Overfitting
- Number of layers
- Number of hidden nodes
- Learning Rate
  - Controls how much of an adjusment you'll make each epoch (const 0<x<1)
![](resources/multilayered_nn/image10.png)
    - too high = underfit (high bias)
    - too low = overfit (high variance)
![](resources/multilayered_nn/variance_bias.png)
- Activation function
- Weight initialization
  - If weights can be reduced to 0, they can deop out and won't affect function. This can be helpful to avoid overfitting
- Stopping criteria (fixed number, epoch, convergence?)

---
## Deep Networks - Large Neural Networks

---
## Convolutional Neural Networks

---
## Ensemble Classifiers

---
## K Means++

### cluster initiation
### furthest first
### probabilistic selection of cluster means
---
## Dimensionality Reduction

### Principal component analysis, Principal components
### Minimize data distance to line
### maximize distance of projected points to origin
### selecting components
### *Visualization of first component, additional components

---
## Overall: highlights of algorithms, *compare and contrast

### NBC

### linear regression
### logistic regression
### decision tree
### knn
### random forest
### neural network
### k means
### boosting
### svm