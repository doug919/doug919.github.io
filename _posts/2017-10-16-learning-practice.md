---
layout: post
title: Hyperparameter Tuning
category: researchnotes
---

The reference [1] has thorough explanations. Here I only note the things that I personally need.

## Gradient Checks

- Compare the analytical and numerical gradients.
\\[
  \frac{d f(x)}{dx} = \frac{f(x+h) - f(x-h)}{2h}
\\]
- use relative error for the comparison
\\[
  \frac{|f'_a - f'_n|}{max(|f'_a|, |f'_n|)}
\\]
errors under 1e-4 usually means it's good.

## Learning Process
![learning]({{"/images/posts/learningrates.jpg" | absolute_url }})
(figure from [1])

- Loss vs. Epoch plots.
  - Low learning rate results in linear progress; high learning rate results in sticking in an overall high loss. progress; high learning rate results in sticking in an overall high loss.
  - If the lines wiggle a lot, it might indicate that the batch size is too small. indicate that the batch size is too small.


- Train/Val Accuracy
  - If the validation accuracy is far lower than training, it is overfitting.
  - If the validation accuracy is close to the training, it means the model is too simple to learn things. You might need to increase the model parameters.

- Ratio of Weight Updates
  - The ratio of the update magnitudes to the value magnitudes (using 2-norm for both) should be around **1e-3**. If it's lower than 1e-3, the learning rate might be too low, and vice versa.

## Identify Incorrect Initialization

Plot activation/gradient histogram for all layers. Should not have any strange distributions (e.g., biasing toward any boundaries).

## First Layer Visualization could be Useful in Computer Vision.

## Hyperparameters
- master-slave threading
- prefer one fixed validation fold
- **random search** rather than grid search (log-scale search)

## Evaluation
- Model Ensembels: average indepdent models' predications could slightly increase the final performance.


## reference
1. http://cs231n.github.io/neural-networks-3/
