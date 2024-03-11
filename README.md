## DENSITY sampler 

Example implementation of the **DENSITY sampler** proposed in ["How to Train Data-Efficient LLMs" (Sachdeva et al., 2024)](https://arxiv.org/abs/2402.01613). 

### The algorithm: Inverse Propensity Sampling (IPS) via Kernel Density Estimation (KDE)

A first pass over the dataset is performed to construct a kernel density estimator (KDE) for the data (the `sketch`):
1. Use `R` independent hash functions to hash the input data (embeddings of training data documents).
2. Use a `sketch` (RxB) to count the number of times each hash value appears for a particular hash function (over the training data).
 
A second pass over the data to compute scores/weights for each data point:
1. Initialize empty list of `scores`
2. Initialize `score=0` for each data point.
3. For a given data point, hash it using the same `R` hash functions.
4. For each hash function, look up the count of the hash value index in the `sketch`. 
5. Add the count (score) of each hash function to `score` to compute a sum of scores for the data point.
6. Average `score` over the `R` hash functions to get the final score for the data point (`score/R`).
7. Append the data point's score to the list of scores.

Finally, calculate the sampling probabilities (inverse propensity): `weights = np.Array(scores) / sum(scores)`

### Implementation uncertainties

* The reference L2-norm LSH implementation for P-stable distributions by Coleman uses a normal distribution with default parameters (mean=0, std=1) to construct the random projection matrix. This leads to hash values that are both negative and postitive. While negative indexing is possible, it might cause issues whenever the ranges of positive and negative indices are large enough to collide (i.e. larger than half the hash range: `abs(hash_value) >= B/2`). One can however tune the value of `r` to make it unlikely fort this to happen.

### References

["How to Train Data-Efficient LLMs" (Sachdeva et al., 2024)](https://arxiv.org/abs/2402.01613)  
["Locality-Sensitive Hashing Scheme Based on p-Stable Distributions" (Datar et al., 2004)](https://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p253-datar.pdf)  
["Implementing LSH Functions in Tensorflow" (Benjamin Coleman)](https://randorithms.com/2022/02/11/tensorflow-lsh-functions.html)
