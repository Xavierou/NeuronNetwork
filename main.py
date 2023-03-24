import numpy as np
import matplotlib.pyplot as plt


def linearRegression(X, Y):
    # Превращаем Х в вектор-столбец для дальнейших манипуляций
    X = X.reshape(-1, 1)
    A = np.concatenate((np.ones(X.size).reshape(-1, 1), X), axis=1)
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), Y)  # w = (A^T * A)^(-1) * A^T * Y
    Y_ = w[0] + w[1] * X.ravel()
    plt.plot(X.ravel(), Y_, Y, 'o')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    x = np.random.randint(100, size=100)
    y = np.random.randint(100, size=100)
    linearRegression(x, y)
