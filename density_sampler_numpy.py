import numpy as np
from tqdm import tqdm

"""
LSH based Density sampling using P-stable distributions

# See How to Train Data-Efficient LLMs
# Algorithm 2: https://arxiv.org/pdf/2402.09668.pdf

# See "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions" section 3.2
# https://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p253-datar.pdf

Further references:
https://github.com/brc7/tensorflow-lsh-functions/blob/main/lsh_functions.py
"""


def lsh_l1(embedding, num_hashes, r, seed=0):
    """
    Locality sensitive hashing for vectors in L1 space.
    Code reference (tensorflow implementation):
    https://github.com/brc7/tensorflow-lsh-functions/blob/42ea9644fe46e6f19cc6a1a34e0198a0b178e3e4/lsh_functions.py#L202-L296
    """
    np.random.seed(seed)
    embed_dim = embedding.shape[1]
    w = np.random.uniform(low=0, high=1, size=(embed_dim, num_hashes))
    w = np.tan(np.pi * (w - 0.5))
    b = np.random.uniform(0, r, size=(1, num_hashes))
    affine_projection = np.dot(embedding, w) + b  # [batch_size, num_hashes]
    hash_values = np.floor(affine_projection).astype(int)
    return hash_values


def lsh_l2(embedding, num_hashes, r, seed=0):
    """
    Locality sensitive hashing for vectors in L2 space.

    Args:
        embedding: Array of shape [batch_size, embed_dim]
        num_hashes: number of hash functions
        r: Scale factor. Can also be interpreted as width of the hash bin
            (chop the real line into equi-width intervals of length r).
            Lower r increases the range of hash values.
    """
    np.random.seed(seed)
    embed_dim = embedding.shape[1]
    w = np.random.normal(size=(embed_dim, num_hashes))  # Random projection matrix
    b = np.random.uniform(low=0, high=r, size=(1, num_hashes))  # bias term
    affine_projection = (np.dot(embedding, w) + b) / r  # [batch_size, num_hashes]
    hash_values = np.floor(affine_projection).astype(int)
    return hash_values


#### Initialize variables and params ####
# Initialize sketch matrix of 0s with shape RxB (1000 x 20000)
R = 1000  # Number of hash functions
B = 20000  # Hash range B

# Create a sketch matrix that acts as a counter for the number of times a hash value is seen
sketch = np.zeros((R, B), dtype=np.int32)

batch_size_random = 500  # Number of random embeddings to hash
batch_size_similar = 500  # Number of identical embeddings to hash (algorithm should return higher weights for these)
embed_dim = 768
r = 0.01  # Chop the real line into equi-width intervals of length r


#### First pass over Dataset D to add counts to the sketch matrix ####

# Fake embeddings
np.random.seed(0)
random_embedding = np.random.normal(size=(batch_size_random, embed_dim))
similar_embedding = np.ones((batch_size_similar, embed_dim))
embedding = np.concatenate([random_embedding, similar_embedding], axis=0)

# Calculate hash values
hash_values = lsh_l2(embedding, num_hashes=R, r=r, seed=0)

# For each observation x in D (each embedding),
# increment a 1 at every row R_i of the sketch matrix
# at the column index B_j (this index is given by the hash value h_i(x))
np.add.at(sketch, (np.arange(R), hash_values), 1)


#### Second pass over Dataset D to calculate the density estimate ####
scores = []
for i in tqdm(range(batch_size_random + batch_size_similar)):
    score = 0
    x = embedding[[i], :]  # 1 x embed_dim
    x_hash = lsh_l2(x, num_hashes=R, r=r, seed=0)  # 1 x R

    for hash_idx in range(R):
        score += sketch[hash_idx, x_hash[0, hash_idx]]

    scores.append(score / R)

weights = np.array(scores) / np.sum(scores)  # S / sum(S) where S is the list of scores

print(
    f"Mean weight for random embeddings: {np.mean(weights[:batch_size_random])}"
    f"\nMean weight for similar embeddings: {np.mean(weights[batch_size_random:])}"
    f"\nSimilar embeddings are on average upweighted by a factor of: {np.mean(weights[batch_size_random:]) / np.mean(weights[:batch_size_random])}"
    f"\ncompared to the random embeddings."
)  # Sampling weights for similar embeddings are higher than for random embeddings (density estimate is higher in embedding space)

# Modify r to increase/decrease the probability of hash collisions
