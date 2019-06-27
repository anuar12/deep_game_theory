# DeepCH Pytorch

Repo contains Pytorch implementation of [Deep Learning for Predicting Human Strategic Behavior](https://papers.nips.cc/paper/6509-deep-learning-for-predicting-human-strategic-behavior)
and Pytorch implementation of pairwise comparison (attentional) layers.

The task is to model human behaviour using deep learning. We have a dataset of human actions in normal-form games on which we can train and test our model.

Previous methods utilized expert-engineered game-theoretic features. Our architecture, comprised of several different modules, is capable of representing most of the game-theoretic features (like maximum values, min-max regret, etc...). Yet, we keep the whole architecture differentiable end-to-end while operating on sets. Specifically, the architecture has properties of being 1) size-invariant and 2) permutation invariant.

Using our pairwise comparison layers, which can also represent attention, we can compare any two given actions by doing n^2 comparisons where n is variable. We can apply this attentional layer to any layer which gives an ability to perform comparisons on rich feature space. It can be applied on feature layers which themselves compute most of game-theoretic features out of the box.
