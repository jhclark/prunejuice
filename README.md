About
=====

This is Jon Clark's implementation of the FastOSCAR regularizer. It roughly uses the proximal gradient method described in Zhong & Kwok, but with AdaGrad instead of FISTA as the initial weight update strategy.

The general idea is to combine a L1 regularizer (which encourages sparsity / zero weights) with a pairwise L_infinity regularizer (which encourages less degrees of freedom / less unique weight magnitudes). This is accomplished through a fairly efficient O(n * log n) proximal step (unlike other pairwise L_infinity regularizers, which tend to be solved with quadratic program solvers, which don't scale to much larger feature sets).

The only dependency is the STL and C++11.

This is a fairly rough crack at it right now. Currently, it just uses dense vectors to avoid additional dependencies. If you're working with very large feature sets (1M+ features), it should only be ~1 hour of work to ensparsen the vectors.


Using
=====

Copy fast_oscar.h into your project and call OscarAdaGradOptimize(...).

See fast_oscar_test.cpp for an example usage.


Building and Testing
====================

```bash
./build.sh
```

This script downloads the Google Test dependency, builds that, builds the FastOSCAR+AdaGrad tests, then runs some simple optimization problems.

References
==========

L. Zhong & J. Kwok, Efficient Sparse Modeling With Automatic Feature Grouping, IEEE Transactions on Neural Networks and Learning Systems. 2012. http://www.cse.ust.hk/~jamesk/papers/tnnls12b.pdf

J. Duchi, E. Hazan, & Y. Singer, Adaptive Subgradient Methods for Online Learning and Stochastic Optimization, Journal of Machine Learning Research, 2011. http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi11.pdf

X. Zeng & M. Figueiredo, Solving OSCAR regularization problems by proximal splitting algorithms, Arxiv, 2013. http://arxiv.org/pdf/1309.6301.pdf (This implementation uses the exact grouping proximity operator, not Zeng's approximate proximity operator)