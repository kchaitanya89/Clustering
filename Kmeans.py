import copy
import csv
import math
import random
import sys

import numpy
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
    # input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/cho.txt', 'r')
    input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/iyer.txt', 'r')
    # input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/k.txt', 'r')

    points = read_file(input_file)

    # number_of_clusters = int(raw_input("Enter the number of clusters: "))
    number_of_clusters = 10

    print number_of_clusters
    # print_list(points)
    # choose the centroids
    centroid_indices = [random.randint(1, len(points)) for n in range(0, number_of_clusters)]

    # centroid_indices = [0, 1, 2, 3, 4]
    # centroid_indices = [0, 1]
    # print centroid_indices

    centroids = [copy.deepcopy(points[n]) for n in centroid_indices]
    # print_list(centroids)

    not_converged = True
    previous_classification = {}
    iter_count = 0

    while not_converged and iter_count < 300:
        current_classification = {}
        for point in points:
            min_centroid_index = -1
            min_dist = sys.float_info.max

            for centroid in centroids:
                dst = distance.euclidean(point.features, centroid.features)
                # dst = math.sqrt(sum([math.pow((i[0] - i[1]), 2) for i in zip(point.features, centroid.features)]))
                if dst <= min_dist:
                    min_dist = dst
                    min_centroid_index = centroids.index(centroid)

            point.cluster_id = min_centroid_index

            try:
                current_classification[min_centroid_index].append(point.features)
            except KeyError:
                current_classification[min_centroid_index] = [point.features]

        # Update Centroids
        for key, val in current_classification.items():
            centroids[key].features = [sum(point) / float(len(val)) for point in zip(*val)]

        if previous_classification == current_classification:
            not_converged = False

        iter_count += 1
        previous_classification = current_classification

    print "Number of iterations ran: " + str(iter_count)

    feature_matrix = []
    cluster_matrix = []

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

    jaccard = float(ss) / float(ss + sd + ds)

    print "jaccard = " + str(jaccard)

    d_matrix = distance_matrix(points, )
    correlation_result = corr(d_matrix, points)
    print "corr = " + str(correlation_result)

    # for key, val in previous_classification.items():
    #     print(str(key) + " - " + str(len(val)))

    feature_matrix = []
    cluster_matrix = []
    lis = []
    for point in points:
        plis = [point.cluster_id]
        lis.append(plis + point.features)
        feature_matrix.append(point.features)
        cluster_matrix.append(point.cluster_id)

    with open("./kmeans_output" + str(number_of_clusters) + ".csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(lis)
    f.close()



def print_list(points):
    for point in points:
        print point


def read_file(input_file):
    points = []
    for line in input_file:
        line_array = [float(word) for word in line.split()]
        points.append(Point(line_array[0], line_array[1], line_array[2:]))

    return points


if __name__ == '__main__':
    main()
