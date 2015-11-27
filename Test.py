import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler


def main():

    df = pd.read_csv('/Users/Chaitanya/PycharmProjects/del/db_output1_3.csv');
    # Split the 1st 4 columns comprising values
    # and the last column that has species
    X = df.ix[:, 1:].values
    y = df.ix[:,0].values


    X_std = StandardScaler().fit_transform(X);  # standardization of data


    pca = PCA(n_components=2)
    pca.fit(X)
    X_reduced = pca.transform(X)
    print "Reduced dataset shape:", X_reduced.shape

    import pylab as pl
    #for plotting
    pl.scatter(X_reduced[:,0],X_reduced[:,1],s=50,c=y)
    plt.show()

    sys.exit(0)

    # Fit the model with X_std and apply the dimensionality reduction on X_std.
    pca = PCA(n_components=2)  # 2 PCA components;
    Y_pca = pca.fit_transform(X_std)

    di = {1: (2, 2)}
    # di = {}
    (a, b) = di[1]

    di[1] = (3, 3)

    print di[1]

    sys.exit(0)

    for i in numpy.arange(0.1, 3, 0.1):
        print i

    sys.exit(0)

    a = numpy.array([[0, 0],
                     [1, 0],
                     [0, 1],
                     [1, 1],
                     [0.5, 0],
                     [0, 0.5],
                     [0.5, 0.5],
                     [2, 2],
                     [2, 3],
                     [3, 2],
                     [3, 3]])

    print a
    z = linkage(a)
    d = dendrogram(z)

    print z
    print d

    sys.exit(0)

    data = np.array(np.random.randint(10, size=(10, 3)))
    results = PCA(data)

    # this will return an array of variance percentages for each component
    print results.fracs

    # this will return a 2d array of the data projected into PCA space
    print results.Y
    sys.exit(0)

    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.plot(range(0, 10))
    ax2 = f2.add_subplot(111)
    ax2.plot(range(10, 20))
    plt.show()

    sys.exit(0)

    matrix = [[1, 2, 3], [4, 5, 6]]
    print matrix[1][2]

    sys.exit(0)

    list_of_lists = [[1], [2]]
    print list_of_lists.pop(0)

    sys.exit(0)

    a = [1, 2, 3]

    for index, item in enumerate(a):
        if len(a) > 1:
            a.pop(len(a) - 1)
        else:
            break

    print a

    sys.exit(0)

    a = [1, 2, 3]
    b = [3, 4, 5]

    print math.sqrt(sum([math.pow((i[0] - i[1]), 2) for i in zip(a, b)]))
    print distance.euclidean(a, b)


    # print [ for j in range(0, len(a[0])) for point in zip(a,b)]


if __name__ == '__main__':
    main()
