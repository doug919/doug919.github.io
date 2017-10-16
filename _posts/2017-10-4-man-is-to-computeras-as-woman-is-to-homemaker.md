---
layout: post
title: Man is to computer programmer as woman is to homemaker? Debiasing word embeddings
category: researchnotes
---

## Paper Notes

> Bolukbasi, Tolga, et al. "Man is to computer programmer as woman is to homemaker? Debiasing word embeddings." Advances in Neural Information Processing Systems. 2016.

[[pdf](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)]


---

### Contributions
- Eliminate gender biases in word embeddings
  - Remove `Receptionist-Female`; gender neutral words
  - Keep `Queen-Female`; generder specific words
- Prove *linear separable* between gender-neutral words and baised words
- Algorithm to "debias"
---

### Examples
> man - woman = compupter_programmer - homemaker
---

### Details
- wordvec on Google News, 300d
- GloVe is tested and described in Appendix
---

### Methods
- Gender occupation stereotype exists!
  - M-Turk
- Analogies
  - `(she, he)` -> `(x, y)``
  - Overall, 72 out of 150 analogies were rated as gender-appropriate by five or more out of 10 crowd-workers, and 29 analogies were rated as exhibiting gender stereotype by five or more crowd-workers
  - The first principal component capture the gender subspace.

    ![Image of Yaktocat](img/gender_pc.png)
  - Gender direction? is it the corresponding eigenvector?
  - DirectBiase=0.8 -> high?
-  Debiasing Algorithm
  - identify gender subspace
  - hard debiasing / soft debiasing
- Human evaluation

---
### Interesting Related Work
1. Zhao, Jieyu, et al. "Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints." arXiv preprint arXiv:1707.09457 (2017).
