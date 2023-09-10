def vr_pca(step_size, epoch_length, X):
    errors_arr = []
    max_eigenvalue = 1
    n, d = X.shape
    w_t = np.random.rand(d)
    w_t = w_t / np.linalg.norm(w_t)
    s = range(60)
    for s1 in s:
        u_t = X.T.dot(X.dot(w_t)) / n
        w = w_t
        for t in range(epoch_length):
            i = np.random.randint(n)
            w_tag = w + step_size * (X[i] * (X[i].T.dot(w) - X[i].T.dot(w_t)) + u_t)
            w = w_tag / np.linalg.norm(w_tag)
        w_t = w
        errors_arr.append(np.log10(1.001 - (np.linalg.norm(X.dot(w)) ** 2 / max_eigenvalue ** 2)))
    import matplotlib.pyplot as plt
    plt.plot(s, errors_arr)
    plt.xlabel('s')
    plt.ylabel('log errors')
    plt.show()
    return w_t


import numpy as np
lamda = 0.05
d = 1000
n = 2000
diag1 = [1, 1 - lamda, 1 - 1.1 * lamda, 1 - 1.2 * lamda, 1 - 1.3 * lamda, 1 - 1.4 * lamda]
mu, sigma = 0, 1  # mean and standard deviation
s = np.random.normal(mu, sigma, d - 6) / d
diag2 = np.concatenate((diag1, s), axis=0)
D = np.zeros((d, d))
a = np.random.randn(n, d)
V, r = np.linalg.qr(a)
print(V.shape)
a = np.random.randn(d, d)
U, r = np.linalg.qr(a)
print(U.shape)
print(V.T.shape)
X = np.dot(U * diag2, V.T)
print(X.shape)
m = n
r_h = (X ** 2).sum() / n
eta = 1 / (r_h * np.sqrt(n))
vr_pca(eta, m, X)
print(X.shape)
