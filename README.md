Code for recently published XML algorithm, GLaS in NeurIPS '19 ([LINK](https://papers.nips.cc/paper/8740-breaking-the-glass-ceiling-for-embedding-based-classifiers-for-large-output-spaces)).

```
@incollection{NIPS2019_8740,
title = {Breaking the Glass Ceiling for Embedding-Based Classifiers for Large Output Spaces},
author = {Guo, Chuan and Mousavi, Ali and Wu, Xiang and Holtmann-Rice, Daniel N and Kale, Satyen and Reddi, Sashank and Kumar, Sanjiv},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {4943--4953},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8740-breaking-the-glass-ceiling-for-embedding-based-classifiers-for-large-output-spaces.pdf}
}
```

# How to run
- Launch jupyter notebook
- Open and run `glas_test.ipynb`

# Code organiation
- `EURLex-4K`: Data directory. All data is downloaded from the [XML Repository](http://manikvarma.org/downloads/XC/XMLRepository.html).
- `logs`: Training logs. This GitHub repository contains training logs for various model combinations on EURLex-4K.
- `utils.py`: Data loader. Contains code relevant for reading data from files, and making batches.
- `glas_test.ipynb`: Main code. Contains all code for GLaS.

# Requirements
- pytorch
- tqdm
- scipy
- xclib (for evaluation) ([LINK](https://github.com/kunaldahiya/pyxclib))
