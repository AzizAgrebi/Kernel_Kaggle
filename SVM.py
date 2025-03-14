import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

class SVM():
    """  
    Usage:
        svm = SVM(C=C)
        svm.fit(kernel(X_train, X_train), y_train)
        svm.predict(kernel(X_test, X_train))
    """

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, K, y):
        K = K.astype(np.float64)
        y = y.astype(np.float64)
        n = K.shape[0]

        P = matrix(K)
        q = matrix(-y)
        G = matrix( np.vstack([np.diag(y),-np.diag(y)]) )
        h = matrix( np.hstack([np.full(n, self.C), np.zeros(n)]) )

        result = solvers.qp(P=P,q=q,G=G,h=h)
        self.alphas = np.squeeze(np.array(result['x']))

    def predict(self, K):
        y = np.dot(K, self.alphas)
        return 2*(y > 0) - 1