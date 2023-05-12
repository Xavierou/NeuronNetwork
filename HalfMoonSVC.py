import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm


if __name__ == '__main__':
    x, y = datasets.make_moons(n_samples=500, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=109)

    clf = svm.SVC(C=10, kernel='rbf', probability=True)
    clf.fit(X_train, y_train)

    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    z_prob = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    plt.figure(figsize=(6, 8), dpi=90)
    plt.contour(xx, yy, z_prob, levels=[0.5], linestyles='--')
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Accent)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.show()
