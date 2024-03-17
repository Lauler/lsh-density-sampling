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
6. Instead of averaging `score` over the `R` hash functions to get the final score for the data point, we normalize the score over the total count of the sketch matrix (`score/(R * N)`). The sum `np.sum(sketch)` is `N * R`.
7. Append the data point's score to the list of scores.

Finally, calculate the sampling probabilities (inverse propensity means taking the reciprocal): 

```python
weights = 1 / np.array(scores)
```

### References

["How to Train Data-Efficient LLMs" (Sachdeva et al., 2024)](https://arxiv.org/abs/2402.01613)  
["Locality-Sensitive Hashing Scheme Based on p-Stable Distributions" (Datar et al., 2004)](https://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p253-datar.pdf)  
["Implementing LSH Functions in Tensorflow" (Benjamin Coleman)](https://randorithms.com/2022/02/11/tensorflow-lsh-functions.html)  
Coleman, Benjamin, et al. ["One-pass diversified sampling with application to terabyte-scale genomic sequence streams." International Conference on Machine Learning. PMLR, 2022.](https://proceedings.mlr.press/v162/coleman22a.html)