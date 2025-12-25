import math
from collections import Counter

# Euclidean distance function
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


# KNN algorithm
def knn(train_data, train_labels, test_point, k):
    distances = []

    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_point)
        distances.append((dist, train_labels[i]))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get k nearest neighbors
    k_neighbors = distances[:k]
    k_labels = [label for _, label in k_neighbors]

    # Majority vote
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


# Example usage
if __name__ == "__main__":
    # Training data
    X_train = [
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 7],
        [8, 6]
    ]

    y_train = ["A", "A", "A", "B", "B", "B"]

    # Test point
    X_test = [5, 5]
    k = 3

    prediction = knn(X_train, y_train, X_test, k)
    print("Predicted class:", prediction)
