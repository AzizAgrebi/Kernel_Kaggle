import numpy as np

from scipy.sparse import coo_matrix
from numba import njit

def preprocess(X):
    tf = {c:i for i,c in enumerate('ATCG')}
    return [np.array([tf[c] for c in x], dtype=np.int32) for x in X]

@njit
def kmer_id(kmer):
    id = np.int32(0)
    for c in kmer:
        id = (id << 2) ^ c     # id = 4 * id + c 
    return id

@njit
def extract_kmers(x, k, m=0):
    # kmer_id(u) is the number in [0, 4**k - 1] associated to u.
    # kmer_ids = [kmer_id(u) for u (k,m)-mer of x]

    kmer_ids = np.empty((len(x) - k + 1) * (1 + m*3*k), np.int32)
    count = 0
    for i in range(len(x) - k + 1):
        id = kmer_id(x[i:i+k])
        kmer_ids[count] = id
        count += 1

        if m == 1:
            for pos in range(k):
                for c in range(4):
                    neigh_id = (id & (~(3 << (pos << 1)))) | (c << (pos << 1))
                    if neigh_id != id:
                        kmer_ids[count] = neigh_id
                        count += 1
    return kmer_ids

def embed(X, k, m):
    kmers = [extract_kmers(x, k, m) for x in X]
    n, d = len(kmers), len(kmers[0])
    data = np.ones(n * d, dtype=np.float32)
    rows = np.repeat(np.arange(n),d)
    cols = np.hstack(kmers)
    return coo_matrix((data, (rows, cols)), shape=(n, 4**k))

def mismatch_kernel(X, k, m, Y=None, normalize=True):
    """
    Compute the (k,m)-mismatch_kernel matrix of (X,Y) for m=0 or m=1.
    Processes the datasets almost instantaneously for m=0 (spectrum kernel) 
    and in less than 10 seconds for m=1 (whatever k).
    X : list[np.array[dtype=np.int32]] can be obtained as X = preprocess(iterable[str]).
    """
    if Y is None:
        X_emb = embed(X, k, m)
        K = (X_emb @ X_emb.T).toarray()
        if normalize:
            temp = np.sqrt(np.diag(K))
            K = (K / temp) / temp[:,None]
        return K
    else:
        X_emb, Y_emb = embed(X, k, m), embed(Y, k, m)
        K = (X_emb @ Y_emb.T).toarray()
        if normalize:
            K /= np.sqrt(np.array(X_emb.power(2).sum(axis=1)))
            K /= np.sqrt(np.array(Y_emb.power(2).sum(axis=1))).flatten()
        return K