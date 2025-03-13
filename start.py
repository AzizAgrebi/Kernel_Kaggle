from utils import SpectrumEmbedding, PolyKernel, KernelLogisticRegression
import pandas as pd
import numpy as np


preds = {}

for k in [0, 1, 2]:
    X_train = pd.read_csv(f"data/Xtr{k}.csv")
    y_train = pd.read_csv(f"data/Ytr{k}.csv")
    X_test = pd.read_csv(f"data/Xte{k}.csv")

    X_train = np.array(X_train["seq"])
    y_train = np.array(y_train["Bound"])
    X_test = np.array(X_test["seq"])

    X_train_embed = SpectrumEmbedding(X=X_train, list_k=[5, 7, 12], train=True)
    X_test_embed = SpectrumEmbedding(X=X_test, list_k=[5, 7, 12], train=False, X_train=X_train)

    kernel = PolyKernel(2)
    model = KernelLogisticRegression(kernel, lamda=1, max_iter=15, preprocessing=None, verbose=False)

    model.fit(X_train_embed, y_train)

    y_pred = model.predict(X_test_embed)
    preds[k] = y_pred

final_pred = pd.DataFrame({
    "Id": pd.RangeIndex(0, 3000), 
    "Bound": np.concat([preds[0].reshape(1000), preds[1].reshape(1000), preds[2].reshape(1000)])}).set_index("Id").to_csv("Yte.csv")