---
layout: post
title: Notes on Gradient Descent Optimization
category: researchnotes
---

## Source
---
> [An overview of gradient descent optimization
algorithms](https://arxiv.org/abs/1609.04747)


## Short summary for each algorithm

Here I summarize few modern gradient descent algorithms as they are so common to be seen in research papers. I do not understand why many researchers don't appreciate the work in this line. However, to me, they do look amazing and intelligent, and that's why I would like to write a note down here.

Most of the content comes from the above source paper, which is the best summary of this sort of algorithms I have seen in recent years. I recommend that you read it.

### Momentum

Accelerate along the slope.

$$\begin{align*}
  v_t &= \gamma v_{t-1} + \eta \nabla_{\theta} J(\theta) \\
  \theta &= \theta - v_t
\end{align*}$$

### Nesterov Accelerated Gradient

Slow down the velocities before hitting the local optimum by looking ahead the estimated update $$\theta - \gamma v_{t-1}$$.

$$\begin{align*}
  v_t &= \gamma v_{t-1} + \eta \nabla_{\theta} J(\theta - \gamma v_{t-1}) \\
  \theta &= \theta - v_t
\end{align*}$$

### AdaGrad

It introduces per-parameter updates or learning rates, which is good for sparse data, since it can run a bigger step for the parameters that are rarely updated, and vice versa. On successful story for this is that *Glove*, a well-known word embedding model, uses this for updating rare words' embedding.

$$\begin{align*}
  \theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
\end{align*}$$

where $$G_t$$ is the sum of squared historical gradients.

### AdaDelta

The weaknesses of AdaGrad are (1) that the learning rate will eventually become infinitesimally small and (2) that the updates based on the gradients and the learning rates are different in "units".

To solve the first problem, it uses a decaying average gradients:
\\[
  G_t = \rho G_{t-1} + (1-\rho) g_t^2
\\]
And for the second problem, they create another variable $$\Delta x_t$$ and store histories of the variable using the same way as the gradients so that it maintains the hypothetical "units". Note that the $$\Delta x_t$$ can also be noted as $$\Delta \theta_t$$ if that makes you feel more compatible to other algorithms.

$$\begin{align*}
\Delta x_t &= - \frac{\sqrt{D_{t-1} + \epsilon}}{\sqrt{G_t + \epsilon}} g_t \\
D_t &= \eta D_{t-1} + (1-\eta) \Delta x_t^2 \\
x_{t+1} &= x_t + \Delta x_t
\end{align*}$$

Note that this method doesn't even need the initial learning rate. It completely depends on the historical $$\Delta x_t$$.

### RMSprop

This is basically the AdaDelta that skips the unit issue.

$$\begin{align*}
G_t &= \rho G_{t-1} + (1-\rho) g_t^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t
\end{align*}$$

### Adam

A mix of AdaGrad and RMSProp. Adam calculates a decaying average of the gradient and the squared gradient, controlled by $$\beta_1$$ and $$\beta_2$$.

$$\begin{align*}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_1 v_{t-1} + (1-\beta_1) g_t^2
\end{align*}$$

where $$m_t$$ is the first moment (mean) and $$v_t$$ is the second moment (uncentered variance). Moreover, to avoid that in the early steps they would bias toward zero, a bias correlation mechanism is adopted.

$$\begin{align*}
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t}
\end{align*}$$

Lastly, the Adam update rule
\\[
  \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\\]

The author also suggests values for each parameter:
- $$\eta$$: 0.1
- $$\beta_1$$: 0.9
- $$\beta_2$$: 0.999
- $$\epsilon$$: 10e-8


## Take home
In modern deep learning frameworks, two commonly suggested optimization algorithms worth trying first are
- SGD + Nesterov
- Adam


## References
1. https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
2. http://cs231n.github.io/neural-networks-3/
2. [An overview of gradient descent optimization
algorithms](https://arxiv.org/abs/1609.04747)
3. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
