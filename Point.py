class Point:
    def __init__(self, point_id, ground_truth, features):
        self.point_id = point_id
        self.ground_truth = ground_truth
        self.features = features
        self.cluster_id = None
        self.centroid_index = False
        self.visited = False
        self.pos = point_id - 1

    def __str__(self):
        return str(self.point_id) + " - " + str(self.ground_truth) + " - " + str(self.features)

    def __repr__(self):
       return str(self.point_id)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.point_id == other.point_id