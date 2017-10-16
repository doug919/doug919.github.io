---
layout: post
title: Notes on Backpropagation with Cross Entropy
category: researchnotes
---

## Overview
---

This note introduces backpropagation for a common neural network, or a
multi-class classifier. Specifically, the network has $$L$$ layers, containing
Rectified Linear Unit (ReLU) activations in hidden layers and Softmax in
the output layer. Cross Entropy is used as the objective function to
measure training loss.

## Notations and Definitions
---

![Notation
Visualization]({{ "/images/posts/notes_backprop.png" | absolute_url }})

The above figure = visualizes the network architecture with notations that you will see in this note. Explanations are listed below:
-   $$L$$ indicates the last layer.

-   $$l$$ indicates a specific layer. It could be equal to $$L$$, i.e.,
    $$l-1=L-1$$, but not always the case.

-   The subscript $$k$$ usually denotes neuron indices in the ouptut layer
    (layer $$l$$).

-   The subscript $$j$$ usually denotes neuron indices in layer $$l-1$$.

-   The subscript $$i$$ usually denotes neuron indices in layer $$l-2$$.

-   $$z_k^l$$ is the weighted sum of activations from the previous layer.
    That is,

      $$ \begin{equation} \label{eq:z} z_k^l = b_k^l + \sum_{j} W_{kj}^{l} a_j^{l-1} \end{equation}$$

-   $$a_k^l$$ refers to neuron activations, \\(a_k^l = f(z_k^l)\\), where $$f(.)$$ is the activation function. In this note, we assume that the last layer uses a softmax \\[a_k^L = softmax(z_k^L) = \frac{e^{z_k^L}}{\sum_{c} e^{z_c^L}}\\]
, and the hidden layers use ReLU
\\[a_k^l = relu(z_k^l) = \max(0, z_k^l)\\]

-   $$t_k$$ is the gold probability for the $$k^{th}$$ neuron in the output
    layer. It is one-hot encoded.

-   $$E$$ is the output error measured on one input example. We use Cross Entropy,
    which is defined as below

    $$
    \begin{equation} \label{eq:cross_entropy}
        E = - \sum_{d} t_d \log a_k^L = - \sum_{d} t_d (z_d^L - \log \sum_c e^{z_c^L})
    \end{equation}
    $$

## Gradient Descent
---

Variants of gradient descent algorithms can be applied to update weight matrices $$W_{kj}^l$$. Here we use the most common update rule, which is
\\(\Delta W \propto -\frac{\partial E}{\partial W}\\). In the simpliest
case, a learning rate $$\epsilon$$ is used to control the step size and we
can calculate the derivatives using Chain Rule, so the update rule can
be written as
\\[\Delta W_{kj}^l = - \epsilon \frac{\partial E}{\partial a_{k}^l} = - \epsilon \frac{\partial E}{\partial a_{k}^l} \frac{\partial a_{k}^l}{\partial z_{k}^l} \frac{\partial z_{k}^l}{\partial W_{kj}^l}\\]

## Backpropagation
---

Backprpagation provide us an elegant way to calculate
$$\frac{\partial E}{\partial a_{k}^l}$$ for each layer using a recursive
definition of $$\delta_k^l$$ and $$\delta_j^{l-1}$$ for the adjacent layers
$$l$$ and $$l-1$$, so that the updates can be calculated, or propagated, in
backward order. We will see how to derive $$\delta_k^l$$ and
$$\delta_j^{l-1}$$ from deriving the updates for layer $$l=L$$ and $$l-1$$.

### Update for the Last Layer

Instead of calculating
$$\frac{\partial E}{\partial a_{k}^L} \frac{\partial a_{k}^L}{\partial z_{k}^L} \frac{\partial z_{k}^L}{\partial W_{kj}^L}$$,
we calculate
$$\frac{\partial E}{\partial z_{k}^L} \frac{\partial z_{k}^L}{\partial W_{kj}^L}$$,
because it simplifies the derivation a lot for Softmax and Cross
Entropy.
\\[
\begin{equation} \label{eq:update1\_1}
    \frac{\partial E}{\partial W_{kj}^L} = \frac{\partial E}{\partial z_{k}^L} \frac{\partial z_{k}^L}{\partial W_{kj}^L}
\end{equation}
\\]
Equation \eqref{eq:cross_entropy} has the definition of the error $$E$$,
and we can calculate its derivative with respect to $$z_k^L$$ as follows:
$$
\begin{align} \label{eq:update1_2}
    \frac{\partial E}{\partial z_{k}^L} &= - \sum_{d} t_d (\mathbb{1}_{d=k} - \frac{1}{\sum_c e^{z_c^L}} e^{z_k^L}) \nonumber \\
    &= - \sum_d t_d (\mathbb{1}_{d=k} - a_k^L) \nonumber \\
    &= \sum_d t_d a_k^L - \sum_d t_d \mathbb{1}_{d=k} \nonumber \\
    &= a_k^L \sum_d t_d - t_k \nonumber \\
    &= a_k^L - t_k
\end{align}
$$
, where the $$\mathbb{1}_{d=k}$$ is an identify function:

$$
\mathbb{1}_{d=k} =
    \begin{cases}
        1  & \quad \text{if } d=k \\
        0  & \quad \text{otherwise }.
    \end{cases}
$$

Then we can define $$\delta_k^L$$ as

$$ \label{eq:update1_3}
\delta_k^L = \frac{\partial E}{\partial z_{k}^L} = a_k^L - t_k
$$

We got the first part of Equation \eqref{eq:update1\_1}, so can move on to
the second part which is $$\frac{\partial z_{k}^L}{\partial W_{kj}^L}$$.
Referring to the definition of Equation \eqref{eq:z}, this is trivial.

$$\label{eq:update1_4}
\frac{\partial z_{k}^L}{\partial W_{kj}^L} = a_j^{L-1}$$

As a result the updates for the weights in the last layer are:

$$\begin{align} \label{eq:update1_5}
    \frac{\partial E}{\partial W_{kj}^L} &= \frac{\partial E}{\partial z_{k}^L} \frac{\partial z_{k}^L}{\partial W_{kj}^L} = \delta_k^L a_j^{L-1}
\end{align}$$

We also need to do a similar derivation for the bias.

$$\begin{align}
 \label{eq:update1_6}
\frac{\partial E}{\partial b_{k}^L} &= \frac{\partial E}{\partial z_{k}^L} \frac{\partial z_{k}^L}{\partial b_{k}^L} = \delta_k^L (1) = \delta_k^L\end{align}$$

### Update for the Second Last Layer

Similarly, Equation \eqref{eq:update2_2} - \eqref{eq:update2_4} derives
each component of Equation \eqref{eq:update2_1}

$$\begin{align} \label{eq:update2_1}
\frac{\partial E}{\partial W_{ji}^{l-1}} &= \frac{\partial E}{\partial a_{j}^{l-1}} \frac{\partial a_{j}^{l-1}}{\partial z_{j}^{l-1}} \frac{\partial z_{j}^{l-1}}{\partial W_{ji}^{l-1}}
\end{align}$$

$$\begin{align}
\frac{\partial E}{\partial a_{j}^{l-1}} &= \sum_k \frac{\partial E}{\partial z_k^l} \frac{\partial z_k^l}{\partial a_j^{l-1}} \nonumber\\
&= \sum_k \delta_k^l W_{kj}^l \label{eq:update2_2}
\end{align}$$

$$\begin{equation} \label{eq:update2_3}
\frac{\partial a_{j}^{l-1}}{\partial z_{j}^{l-1}} = f'(z_j^{l-1})
\end{equation}$$

$$\begin{equation} \label{eq:update2_4}
\frac{\partial z_{j}^{l-1}}{\partial W_{ji}^{l-1}} = a_i^{l-2}
\end{equation}$$

Combining all together, we get

$$\begin{align}
 \label{eq:update2_5}
\frac{\partial E}{\partial W_{ji}^{l-1}} &= a_i^{l-2} f'(z_j^{l-1}) \sum_k \delta_k^l W_{kj}^l .\end{align}$$

We can define $$\label{eq:update2_6}
\delta_j^{l-1} = \frac{\partial E}{\partial a_{j}^{l-1}} \frac{\partial a_{j}^{l-1}}{\partial z_{j}^{l-1}} = f'(z_j^{l-1}) \sum_k \delta_k^l W_{kj}^l$$

and re-write Equation \eqref{eq:update2\_5}

$$\begin{align} \label{eq:update2_7}
\frac{\partial E}{\partial W_{ji}^{l-1}} &= \delta_j^{l-1} a_i^{l-2} .\end{align}$$

For the bias

$$\begin{align} \label{eq:update2_8}
\frac{\partial E}{\partial b_{j}^{l-1}} = \frac{\partial E}{\partial a_{j}^{l-1}} \frac{\partial a_{j}^{l-1}}{\partial z_{j}^{l-1}} \frac{\partial z_{j}^{l-1}}{\partial b_{j}^{l-1}} = \delta_j^{l-1} (1) = \delta_j^{l-1} .\end{align}$$

## Backpropagation Summary
---

To this point, we got all the derivatives we need to update our specific
neural network (the one with ReLU activation, softmax output, and
cross-entropy error), and they can be applied to arbitrary number of
layers. In fact, Backpropagation can be generalized and used with any
activations and objectives. It is summarized in the following four
equations:

$$\begin{align*}
 \label{eq:summary1}
\delta_k^l &= \frac{\partial E}{\partial z_k^l} \\
\delta_j^{l-1} &= f'(z_j^{l-1}) \sum_k \delta_k^{l} W_{kj}^l \\
\frac{\partial E}{\partial W_{kj}^l} &= \delta_k^l a_j^{l-1} \\
\frac{\partial E}{\partial b_{k}^l} &= \delta_k^{l}\end{align*}$$

## Numerically Stable Softmax
---

This is a practical implementation issue. Calculating the exponentials
in Softmax is numerically unstable, since the values could be extremely
large. We can do a small trick by introducing a constant $$C$$ to mitigate
such problem.

$$\begin{align*}
softmax(x_i) &= \frac{e^{x_i}}{\sum_j e^{x_j}} \\
&= \frac{C e^{x_i}}{C \sum_j e^{x_j}} \\
&= \frac{e^{x_i+\log C}}{\sum_j e^{x_j + log C}}
\end{align*}$$

A common choice for the constant is $$log C = -\max_{j} x_j$$.
