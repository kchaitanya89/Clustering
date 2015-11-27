import csv
import math
import sys
import time

import numpy
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance

from Point import Point


def distance_matrix(points):
    d_mat = numpy.matrix(numpy.zeros((len(points), len(points))))
    for i in points:
        for j in points:
            d_mat[i.point_id - 1, j.point_id - 1] = distance.euclidean(i.features, j.features)
    return d_mat


def corr(dist_matrix, points):
    clustering_result = numpy.matrix(numpy.zeros((len(points), len(points))))
    for i in points:
        for j in points:
            if (i.cluster_id == j.cluster_id):
                clustering_result[i.point_id - 1, j.point_id - 1] = 1
    mean, cluster_mean = numpy.mean(dist_matrix), numpy.mean(clustering_result)
    numer, denom1, denom2 = 0, 0, 0
    for i in points:
        for j in points:
            numer = numer + (dist_matrix[i.point_id - 1, j.point_id - 1] - mean) * (
                clustering_result[i.point_id - 1, j.point_id - 1] - cluster_mean)
            denom1 += math.pow((dist_matrix[i.point_id - 1, j.point_id - 1] - mean), 2)
            denom2 += math.pow(clustering_result[i.point_id - 1, j.point_id - 1] - cluster_mean, 2)
    correlation = float(numer) / (math.sqrt(denom1) * math.sqrt(denom2))
    return correlation


def main():
    start_time = time.time()

    # input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/dataset1.txt', 'r')
    input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/iyer.txt', 'r')
    # input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/cho.txt', 'r')
    # input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/simple_h2.txt', 'r')
    # input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/simple_hierar.txt', 'r')
    numberOfClusters = 6
    points = read_file(input_file)

    x = []
    for point in points:
        x.append(point.features)

    groups = [[point] for point in points]

    matrix = [];
    for i, p1 in enumerate(points):
        row = []
        for j, p2 in enumerate(points):
            if (i != j):
                row.append(distance.euclidean(p1.features, p2.features))
            else:
                row.append(sys.float_info.max)
        matrix.append(row)

    # print 'distance matrix is '
    # print numpy.asmatrix(matrix)

    linkage_matrix = []

    copy_group = groups[:]

    print 'running the algo...'
    while len(groups) > numberOfClusters:
        outer_to_merge = -1
        inner_to_merge = -1
        min = sys.float_info.max

        for i, outer in enumerate(groups):
            for j, inner in enumerate(groups):
                if i != j:
                    dst = min_distance_in_groups(outer, inner, matrix)
                    if dst <= min:
                        min = dst
                        outer_to_merge = i
                        inner_to_merge = j

        inner_group = groups[inner_to_merge]
        outer_group = groups[outer_to_merge]
        groups.append(inner_group + outer_group)
        copy_group.append(inner_group + outer_group)

        if inner_to_merge > outer_to_merge:
            groups.pop(outer_to_merge)
            groups.pop(inner_to_merge - 1)
            linkage_matrix.append([copy_group.index(outer_group), copy_group.index(inner_group), min,
                                   len(inner_group) + len(outer_group)])
        else:
            groups.pop(inner_to_merge)
            groups.pop(outer_to_merge - 1)
            linkage_matrix.append([copy_group.index(inner_group), copy_group.index(outer_group), min,
                                   len(inner_group) + len(outer_group)])

        print('merged{'),
        for g in inner_group:
            print(str(g.point_id) + ','),
        print('} and {'),
        for g in outer_group:
            print(str(g.point_id) + ','),
        print('}')

    cluster_id = 1

    if numberOfClusters == 1:
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(linkage_matrix, leaf_rotation=90., leaf_font_size=8., )
        plt.show()

    print "total number of groups = " + str(len(groups))
    for group in groups:
        print "for group of len " + str(len(group))
        print cluster_id

        for point in group:
            point.cluster_id = cluster_id
        cluster_id += 1

    feature_matrix = []
    cluster_matrix = []
    lis = []
    for point in points:
        plis = [point.cluster_id]
        lis.append(plis + point.features)
        feature_matrix.append(point.features)
        cluster_matrix.append(point.cluster_id)

    with open("./hier_output.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(lis)
    f.close()

    print "Groups ="
    for point in points:
        print str(point.point_id) + "=" + str(point.cluster_id)

    clustering, ground_truth_matrix = numpy.matrix(numpy.zeros((len(points), len(points)))), numpy.matrix(
        numpy.zeros((len(points), len(points))))
    ss, sd, ds, dd = 0, 0, 0, 0
    for i in range(len(points)):
        for j in range(len(points)):
            if points[i].ground_truth == points[j].ground_truth:
                ground_truth_matrix[i, j] = 1
            if points[i].cluster_id == points[j].cluster_id:
                clustering[i, j] = 1
    for i in range(len(points)):
        for j in range(len(points)):
            if ground_truth_matrix[i, j] == 1 and clustering[i, j] == 1:
                ss += 1
            if ground_truth_matrix[i, j] == 1 and clustering[i, j] == 0:
                sd += 1
            if ground_truth_matrix[i, j] == 0 and clustering[i, j] == 1:
                ds += 1
            if ground_truth_matrix[i, j] == 0 and clustering[i, j] == 0:
                dd += 1

    rand = float(ss + dd) / float(ss + sd + ds + dd)
    jaccard = float(ss) / float(ss + sd + ds)
    print "jaccard = " + str(jaccard)

    d_matrix = distance_matrix(points, )
    correlation_result = corr(d_matrix, points)
    print "corr = " + str(correlation_result)

    # for i in numpy.arange(1.7,1.9,0.01):
    #     clusters = fcluster(linkage_matrix, i, criterion='distance')
    #     print("clusters at distance " + str(i))
    #     print clusters

    print("--- %s seconds ---" % (time.time() - start_time))


def min_distance_in_groups(outer, inner, matrix):
    min = sys.float_info.max
    for p1 in outer:
        for p2 in inner:
            dst = matrix[int(p1.point_id) - 1][int(p2.point_id) - 1]
            # dst = distance.euclidean(p1.features, p2.features)
            # dst = math.sqrt(sum([math.pow((i[0] - i[1]), 2) for i in zip(p1.features, p1.features)]))
            if dst <= min:
                min = dst
    return min


def read_file(input_file):
    points = []
    for line in input_file:
        line_array = [float(word) for word in line.split()]
        points.append(Point(line_array[0], line_array[1], line_array[2:]))
    return points


if __name__ == '__main__':
    main()
