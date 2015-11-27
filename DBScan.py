import csv
import math
import time

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


def get_neighbors(eps, point, points):
    neighbors = []
    for point_considered in points:
        if point != point_considered:
            dst = distance.euclidean(point.features, point_considered.features)
            if dst <= eps:
                neighbors.append(point_considered)
    return neighbors


def expand_cluster(eps, min_pts, point, neighbors, cluster_id, points):
    for idx, neighbor_point in enumerate(neighbors):
        if not neighbor_point.visited:
            neighbor_point.visited = True
            new_neighbors = get_neighbors(eps, neighbor_point, points)
            if len(new_neighbors) > min_pts:
                neighbors += new_neighbors

        if neighbor_point.cluster_id == None or neighbor_point.cluster_id == -1:
            neighbor_point.cluster_id = cluster_id


def main():
    start_time = time.time()

    input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/iyer.txt', 'r')
    # input_file = open('/Users/Chaitanya/Desktop/CSE_601/Project2/datasets/simple_hierar.txt', 'r')

    eps = 1
    min_pts = 3
    cluster_id = 0

    print 'eps ='+str(eps)+' min_pts ='+str(min_pts)
    points = read_file(input_file)
    for point in points:
        if not point.visited:
            point.visited = True
            neighbors = get_neighbors(eps, point, points)

            if len(neighbors) > min_pts:
                cluster_id += 1
                expand_cluster(eps, min_pts, point, neighbors, cluster_id, points)
            else:
                point.cluster_id = -1

    feature_matrix = []
    cluster_matrix = []
    lis = []
    for point in points:
        plis = [point.cluster_id]
        lis.append(plis + point.features)
        feature_matrix.append(point.features)
        cluster_matrix.append(point.cluster_id)

    with open("./db_output" + str(eps) + "_" + str(min_pts) + ".csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(lis)
    f.close()

    counter = {}
    for point in points:
        if point.cluster_id == None:
            print point.point_id

        # print str(point.point_id) + " : " + str(point.ground_truth) + " : " + str(point.cluster_id)
        try:
            counter[point.cluster_id] += 1
        except KeyError:
            counter[point.cluster_id] = 1

    for k, val in counter.items():
        print str(k) + "=" + str(val)

    jaccard = 0.
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

    d_matrix = distance_matrix(points)
    correlation_result = corr(d_matrix, points)
    print "corr = " + str(correlation_result)


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


def read_file(input_file):
    points = []
    for line in input_file:
        line_array = [float(word) for word in line.split()]
        points.append(Point(line_array[0], line_array[1], line_array[2:]))
    return points


if __name__ == '__main__':
    main()
