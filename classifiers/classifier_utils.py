############################################################################
# Name: Liam Navarre, Connor Deide
# Class: CPSC 322, Fall 2022
# Final Project
# 12/14/2022
#
# Description: This file contains utility functions called by the
# classifiers in the classifiers.py file.
############################################################################

import numpy as np
import operator

def compute_euclidean_distance(v1, v2):
    """
    Function computes the distance between two values using the euclidean
    distance algorithm
    Args:
        v1 (float): point one
        v2 (float): point two
    Returns:
        float: the distance value between the two points
    """
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def find_majority_1D(classifications):
    """
    Function returns the most common classification value from a 1D list
    Args:
        classifications (list): list of classification values from y_train
    Returns:
        obj: the most common classification from the list
    """
    # Build dictionary containing frequency of each classification
    y_predicted = ""
    class_frequency = {}
    for value in classifications:
        if value in class_frequency:
            class_frequency[value] += 1
        else:
            class_frequency[value] = 1

    # Check for a tie among all classifications
    check_dict = list(class_frequency.values())[0]
    check = True
    for key in class_frequency:
        if class_frequency[key] != check_dict:
            check = False
            break
    if check:
        return None
    # If here, then there was no tie
    y_predicted = max(class_frequency, key=class_frequency.get)

    return y_predicted

def find_majority_2D(classifications):
    """
    Function returns a list of the most common classification values from each row
    in a 2D list
    Args:
        classifications (list of list): 2D list of classification values from y_train
    Returns:
        list: the most common classification from each row of classifications in 
        the list
    """
    y_predicted = []
    for row in classifications:
        class_frequency = {}
        for value in row:
            if value in class_frequency:
                class_frequency[value] += 1
            else:
                class_frequency[value] = 1
        y_predicted.append(max(class_frequency, key=class_frequency.get))

    return y_predicted

def get_indices_and_dists(MyKNeighborsClassifier, X_test, categorical):
    """
    Function creates a 2D list of tuples (index, distance) based off of the
    distances computed between and instance of X_train and X_test
    Args:
        X_test (list of list): 2D list of point values to be used in computing
        distances
    Returns:
        list of list: 2D list of tuples (index, distance)
    """
    row_indexes_dists = []
    for test_point in X_test:
        index_distances = []
        for i, train_point in enumerate(MyKNeighborsClassifier.X_train):
            if categorical:
                if test_point == train_point:
                    distance = 0
                elif test_point != train_point:
                    distance = 1
            else:
                distance = round(compute_euclidean_distance(\
                    test_point, train_point), 3)
            index_distances.append((i, distance))
        row_indexes_dists.append(index_distances)
    return row_indexes_dists

def get_kclosest_neighbors(knn_clf, row_indexes_dists):
    """
    Function finds the k closest neighbors for each list in the 2D list
    row_indexes_dists by finding the items with the shortest distances
    Args:
        knn_clf (MyKNeighborsClassifier): MyKNeighborsClassifier object used to
        obtain the n_neighbors attribute
        row_indexes_dists (list of list): 2D list of tuples (index, distance) used
        to find the k closest neighbors
    Returns:
        list of list: 2D list of tuples
    """
    total_neighbors = []
    for index_distances in row_indexes_dists:
        index_distances.sort(key=operator.itemgetter(-1))
        total_neighbors.append(index_distances[:knn_clf.n_neighbors])
    return total_neighbors

def find_unique_attributes(X_train):
    """
    This function finds all of the unique attribute values in each column
    within the dataset ignoring duplicates
    Args:
        X_train (list of list): The dataset
    Returns:
        list of list: A 2D list where eahc nested list represents a column
        and the values within each nested list reprsent the unique attribute
        values within the column
    """

    attributes = []
    for idx in range(len(X_train[0])):
        # Want a new list for each row
        row_attributes = []
        for row in X_train:
            if row[idx] not in row_attributes:
                row_attributes.append(row[idx])
        attributes.append(row_attributes)

    return attributes

def compute_posteriors(X_train, y_train, classifications, classifications_count):
    # Build posteriors dictionary architecture
    unique_attributes = find_unique_attributes(X_train)
    posteriors = {}
    for classification in classifications:
        attribute_dict = {}
        for attribute, col_attributes in enumerate(unique_attributes):
            value_dict = {}
            for value in col_attributes:
                value_dict[value] = 0
            attribute_dict[attribute] = value_dict
        posteriors[classification] = attribute_dict

    # Compute posteriors
    for class_idx, classification in enumerate(classifications):
        for attribute, col_attributes in enumerate(unique_attributes):
            for value in col_attributes:
                y_idx = 0
                for row in X_train:
                    if row[attribute] == value and y_train[y_idx] == classification:
                        posteriors[classification][attribute][value] += 1
                    y_idx += 1
                posteriors[classification][attribute][value] /= classifications_count[class_idx]
    return posteriors

def find_naive_prediction(predictions, classifications, priors):
    """
    This function is called within the the Naive Bayes Classifiers predict() function.
    Given the prediction options, it chooses the correct prediction according to the
    algorithm. First looks for the largest prediction, if there is a tie it then looks
    for the largest prior, if there is a tie it then chooses randomly
    Args:
        predictions (dictionary): A dictionary of predictions to choose from
        classifications (list): A list of all the classifications present in the dataset.
            Used for accessing the predictions and priors dictionaries
        priors (dictionary): A dictionary of prior values to choose from
    Returns:
        obj: The prediction value chosen from the predictions
    """

    # Choose largest prediction value
    max_class_value = -1    # -1 instead of 0 to avoid falsely finding a posterior match
    for classification in classifications:
        if predictions[classification] > max_class_value:
            max_class_value = predictions[classification]
            max_class = classification
        elif predictions[classification] == max_class_value:
            # Found a tie, choose prediction w/ the largest prior
            max_prior_value = -1
            for _ in range(len(priors)):
                if priors[classification] > max_prior_value:
                    max_prior_value = classification
                    max_prior = classification
                elif priors[classification] == max_prior_value:
                    # Found a tie, flip a coin
                    rand_idx = np.random.randint(0, len(classifications))
                    return classifications[rand_idx]
            return max_prior
    return max_class

