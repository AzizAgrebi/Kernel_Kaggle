import numpy as np 
import pandas as pd
from tqdm import tqdm
from numba import njit


def Kmer(X, k):
    res = set()
    for x_i in X:
        for j in range(len(x_i) - k + 1):
            res.update([x_i[j: j  + k]])
    return list(res)

def SpectrumEmbedding(X, list_k, train=True, X_train=None):
    kmers = set()
    for k in list_k:
        if train:
            kmers.update(Kmer(X, k=k))
        else:
            kmers.update(Kmer(X_train, k=k))

    X_df = pd.DataFrame(X, columns=["seq"])

    res = []
    for kmer in tqdm(list(kmers)):
        res.append(X_df["seq"].str.count(kmer).values)

    return np.transpose(np.array(res, dtype=np.uint8))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class KernelLogisticRegression:
    def __init__(self, kernel, preprocessing=None, normalization=None,
                 dropout=None, maxi=False, lamda=1, max_iter=15, verbose=True):
        """Kernel Logistic Regression with gradient descent optimization."""

        self.kernel = kernel
        self.preprocessing = preprocessing
        self.normalization = normalization
        self.dropout = dropout
        self.maxi = maxi
        self.lamda = lamda
        self.max_iter = max_iter
        self.verbose = verbose

        # History tracking
        self.alpha_history = []
        self.loss_history = []

    def _apply_kernel(self, X1, X2):
        """Computes the kernel function."""
        return self.kernel.call(X1, X2)

    def _preprocess_kernel(self, K, K_test=None):
        """Applies preprocessing and normalization to the kernel matrix."""
        if self.preprocessing:
            K = self.preprocessing(K)
        if self.normalization:
            if K_test is not None:
                K_test = self.kernel.call(K_test, K_test)
                K_test = self.preprocessing(K_test)
                return self.normalization(K, K_train=self.K_train, K_test=K_test)
            K = self.normalization(K)
        return K

    def _initialize_labels(self, labels):
        """Maps labels to {-1, 1} and stores mappings."""
        self.label_map = {min(labels): -1, max(labels): 1}
        self.inverse_label_map = {-1: min(labels), 1: max(labels)}
        return np.array([self.label_map[y] for y in labels]).reshape(-1, 1)

    def fit(self, X_train, y_train, alpha_init=None):
        """Fits the model using gradient descent."""
        self.X_train = X_train
        self.y_train = self._initialize_labels(y_train)

        # Compute kernel matrix
        self.K_train = self._apply_kernel(X_train, X_train)
        self.K_train = self._preprocess_kernel(self.K_train)

        # Handle dropout
        if self.maxi:
            self.y_train = self.y_train[::2]

        # Initialize alpha
        self.alpha = np.zeros((self.K_train.shape[1], 1)) if alpha_init is None else alpha_init

        self._gradient_descent()

    def predict(self, X_test, avg_size=3):
        """Predicts labels for new data."""
        K_test = self._apply_kernel(self.X_train, X_test)
        K_test = self._preprocess_kernel(K_test)

        alpha_avg = np.mean(self.alpha_history[-avg_size:], axis=0).reshape(-1)
        y_pred = np.where(sigmoid(alpha_avg.T @ K_test) >= 0.5, 1, -1)

        return np.array([self.inverse_label_map[y] for y in y_pred]).reshape(-1, 1)

    def predict_proba(self, X_test, avg_size=3):
        """Predicts class probabilities."""
        K_test = self._apply_kernel(self.X_train, X_test)
        alpha_avg = np.mean(self.alpha_history[-avg_size:], axis=0).reshape(-1)
        return sigmoid(alpha_avg.T @ K_test)

    def score(self, X_test, y_test):
        """Computes accuracy of the model."""
        y_pred = self.predict(X_test)

        if self.maxi:
            y_test = y_test[::2]

        return np.mean(y_pred == y_test.reshape(-1, 1))

    def _compute_loss(self):
        """Computes the loss function value."""
        K_grad = self.dropout(self.K_train) if self.dropout else self.K_train
        loss = np.log(1 + np.exp(-self.y_train * (K_grad @ self.alpha)))
        return np.mean(loss) + (self.lamda / 2) * (self.alpha.T @ K_grad @ self.alpha)

    def _compute_gradient(self):
        """Computes the gradient of the loss function."""
        n = self.K_train.shape[0]
        K_grad = self.dropout(self.K_train) if self.dropout else self.K_train
        grad = (1 / n) * (K_grad @ np.diagflat(self.P) @ self.y_train)
        return grad + self.lamda * (K_grad @ self.alpha)

    def _gradient_descent(self):
        """Performs gradient descent optimization."""
        n = self.K_train.shape[0]
        self.P = np.random.rand(n, 1) + 1e-5

        for i in range(self.max_iter):
            K_grad = self.dropout(self.K_train) if self.dropout else self.K_train

            self.m = K_grad @ self.alpha
            self.P = -sigmoid(-self.y_train * self.m)
            self.W = sigmoid(self.m) * sigmoid(-self.m)
            self.z = self.m - (self.y_train * self.P / self.W)

            W_diag = np.diagflat(self.W)
            inv_term = np.linalg.inv(W_diag @ K_grad + (n * self.lamda + 1e-5) * np.eye(n))
            self.alpha = inv_term @ (W_diag @ self.z)

            self.alpha_history.append(self.alpha.reshape(-1))
            self.loss_history.append(self._compute_loss())

            if self.verbose:
                print(f"Iteration {i + 1}, Loss: {self.loss_history[-1]:.6f}")

@njit
def linear_kernel(X, Y):
    return np.dot(X.astype(np.float64), Y.T.astype(np.float64))

class PolyKernel():
    def __init__(self, k=2):
        self.k = k
    
    def call(self, X, Y):
        return linear_kernel(X, Y)**self.k