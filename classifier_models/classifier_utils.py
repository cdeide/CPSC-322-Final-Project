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

def shuffle_data(random_state, X, y=None):
    """
    This function shuffles two parallel lists randomly while keeping them parellel.
    If y is none, only shuffle X

    Args:
        random_state (int): an integer used for seeding np.random
        X (list of list): A 2D list of values to be sorted parallel to y
        y (list, optional): A list of values to be sorted parallel to X. Defaults to None.

    Returns:
        X and y: returns the now shuffled parellel lists
    """

    if random_state is not None:
        np.random.seed(random_state)

    # shuffle indexes
    for i in range(len(X)):
        rand_idx = np.random.randint(len(X))
        X[i], X[rand_idx] = X[rand_idx], X[i]
        if y is not None:
            y[i], y[rand_idx] = y[rand_idx], y[i]

    return X, y

def split_data(n_splits, X):
    """
    Function splits the dataset X as evenly as possible into n partitions

    Args:
        n_splits (int): number of partitions to split data into
        X (list of list): dataset that needs to be split up

    Returns:
        list of list: A 2D list with lists of split data from X
    """

    splits = []
    for _ in range(n_splits):
        splits.append([])
    idx = 0
    for i in range(len(X)):
        splits[idx].append(i)
        # Check for looping back around
        if idx == n_splits - 1:
            idx = 0
        else:
            idx += 1

    return splits

def split_stratified_data(n_splits, group_indices):
    """
    Function splits the stratified dataset as evenly as possible into n partitions
    Differs from split_data as the original dataset has already been split into
    groups based on classification

    Args:
        n_splits (int): number of partitions to split data into
        group_indices (list of list): each row in this 2D list is a list of all of the
        indexes with corresponding attributes

    Returns:
        list of list: A 2D list with lists of stratified split data from X
    """

    splits = []
    for _ in range(n_splits):
        splits.append([])
    idx = 0
    for group in group_indices:
        for i in range(len(group)):
            splits[idx].append(group[i])
            # Check for looping back around
            if idx == n_splits - 1:
                idx = 0
            else:
                idx += 1

    return splits

def fold_data(n_splits, splits):
    """
    Function 'folds' the data into k tuples. Each tuple has a test set and a training set

    Args:
        n_splits (int): the number of splits in the lists splits
        splits (list of list): A 2D list, each row is a split from the dataset that needs to be folded

    Returns:
        list of list: a list of tuples, each with a test set and train set
    """

    folds = []
    for test_idx in range(n_splits):
        train_set = []
        for idx in range(n_splits):
            if idx == test_idx:
                test_set = splits[idx]
            elif idx != test_idx:
                train_set += splits[idx]
        folds.append((train_set, test_set))
        del train_set
        del test_set
    
    return folds

def group_by_index(y):
    """
    Function creates a list for each classification present in y.
    Then groups the indexes of the elements into their corresponding
    classification

    Args:
        y (list): listof classification values

    Returns:
        list of list: A 2D list where each list is a group of indexes with the
        same classification value
    """

    groups = []
    classes = []
    for i in range(len(y)):
        if y[i] not in classes:
            classes.append(y[i])
            groups.append([])

    for i in range(len(y)):
        for j, classification in enumerate(classes):
            if y[i] == classification:
                groups[j].append(i)
                break

    return groups

def group_by_classification(y):
    """
    Function builds a list of classifications. One for every unique classification
    in the given list. (Gets rid of dups). It also counts how many times each unique attribute
    occurs and builds a list parralel to the classification list
    Args:
        y (list): list of classifications
    Returns:
        list: list of one of each unique classification within y
    """

    classifications = []
    classifications_count = []
    for i in range(len(y)):
        if y[i] not in classifications:
            classifications.append(y[i])
            classifications_count.append(0)

    for i, classification in enumerate(classifications):
        for j in range(len(y)):
            if classification == y[j]:
                classifications_count[i] += 1

    return classifications_count, classifications

