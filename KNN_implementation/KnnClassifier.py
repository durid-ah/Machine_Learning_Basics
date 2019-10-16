import numpy as np


class KnnClassifier:

    def euclidean_distance(self, p1, p2):
        if p2.ndim == 1:  # handle one point
            dist = (p1 - p2)**2
            dist = np.sum(dist)
        else:  # handle multiple points
            dist = (p1 - p2[:, :])**2  # subtracting p2 points from p1 and squaring them
            dist = np.sum(dist, axis=1)  # summing the rows
        dist = np.sqrt(dist)  # square root of every entry
        return dist

    def classify(self, k, class_list, new_data, data_set):
        result_set = []

        for entry in new_data:
            distance_list = self.euclidean_distance(entry, data_set)
            sorted_indices = np.argsort(distance_list, axis=0)
            first_k_indices = sorted_indices[:k]
            array_size = max(class_list) + 1
            count_list = np.zeros(array_size)
            for i in first_k_indices:
                count_list[class_list[i]] += 1
            max_class = None
            max_count = 0
            for i in range(len(count_list)):
                if count_list[i] > max_count:
                    max_count = count_list[i]
                    max_class = i
            result_set.append(max_class)
        return result_set
