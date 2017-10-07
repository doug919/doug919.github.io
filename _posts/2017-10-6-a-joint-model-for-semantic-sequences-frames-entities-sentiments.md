---
layout: post
title: A Joint Model for Semantic Sequences Frames, Entities, Sentiments
category: papernotes
---

# Paper Notes
---
> Peng, Haoruo, Snigdha Chaturvedi, and Dan Roth. "A Joint Model for Semantic Sequences: Frames, Entities, Sentiments." Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017). 2017.

[[pdf](https://aclanthology.coli.uni-saarland.de/pdf/K/K17/K17-1019.pdf)]

## Comments
- Basically, they try to include plentiful event features to better model event semantics, which is unbelievably similar to what we did. The features they use include verb types, sentiments, NER tyeps (essentially the animacy), coreference (new or old).
- To build the event chains, event though they didn't mention too much in their paper, they simply use the position where predicates appear in documents.
- For verb types, they look up the Framenet, which can be done naturally based on their toolkit.
- They include **discourse markers**, such as however and but, as predicates, which is an interesting way to model turning points.
- When it comes to evaluation, the results are amazing, especially for RocStory and Discourse Sense, unlike what I did. Maybe I should survey some open-source projects for those tasks to find ways to improve my implementation.


## Contributions
---
- model design for training event embeddings
- improve discourse sense classification
- state-of-the-art (unsupervised) for RocStory


## Methods
---

##### Event Representations
(from their paper)

![fes_p1]({{ "/images/posts/fes_f1.png" | absolute_url }})


##### frame, entity, sentiment embeddings
\\[r_{RES} = e_f + W_e r_e + W_s r_s\\]

##### log-bilinear
\\[u(c(FES_t)) = \sum_{c \in c(FES_t)} q_i \odot v'(c_i)\\]
\\[p(FES_t | c(FES_t)) = \frac{exp(v(FES_t) \cdot u(c(FES_t)) + b(FES_t) )}{\sum_{FES \in V} exp(v(FES) \cdot u(c(FES)) + b(FES) }\\]

##### Objective
maximize the sequence probability \\(\prod_t p(FES_t|c(FES_t))\\)

---
## Evaluation
- Perplexity
- Narrative Cloze Test
- Discourse sense classification
- **RocStory**
  - The way they compute the conditional probability is good.
  \\[p(s_5 | C) = p(r_{FES_k} | r_{FES_{k-1}, ..., r_{FES_{k-t}}})\\]
  where $$t$$ is a hyperparameter. They report the results of using single highest probability and majority votes.
  - Question: how to pick $$FES_k$$ if there are more than one $$FES$$ in $$S_5$$

## Datasets
- NYT
- RocStory


## Interesting Related Work
- Kordjamshidi, Yangqiu Song Haoruo Peng Parisa, and Mark Sammons Dan Roth. "Improving a Pipeline Architecture for Shallow Discourse Parsing." CoNLL 2015 (2015): 78.
  - The base system to include event scores for discourse sense classifications.
- Peng, Haoruo, and Dan Roth. "Two discourse driven language models for semantics." arXiv preprint arXiv:1606.05679 (2016).
