############################################################################
# Name: Liam Navarre, Connor Deide
# Class: CPSC 322, Fall 2022
# Final Project
# 12/14/2022
#
# Description: This file contains different classification models we have
# built over the duration of the semester. Classifiers include a Dummy
# Classifier, KNeighbors Classifier, Naive Bayes Classifier, and a
# Random Forest Classifier.
############################################################################

from classifiers import classifiers_utils

##############################################################
# DUMMY CLASSIFIER
##############################################################
class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # Find the most frequent class label
        self.most_common_label = classifiers_utils.find_majority_1D(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.most_common_label is None:
            return None
        else:
            y_predicted = []
            for _ in range(len(X_test)):
                y_predicted.append(self.most_common_label)
        return y_predicted


##############################################################
# KNEIGHBORS CLASSIFIER
##############################################################
class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        # Check if X contains categorical attributes
        if type(X_test[0][0]) == str:
            categorical = True
        else:
            categorical = False

        # Get indices and distances
        row_indexes_dists = classifiers_utils.get_indices_and_dists(self, X_test, categorical)
        # Get the k_closest neighbors indexes and values for each
        # row within total distances
        total_neighbors = classifiers_utils.get_kclosest_neighbors(self, row_indexes_dists)
        # Create neighbor_indices and distances from total_neighbors
        neighbor_indices = []
        distances = []
        for row in total_neighbors:
            row_of_neighbor_indices = []
            row_of_distances = []
            for index_distance in row:
                row_of_neighbor_indices.append(index_distance[0])
                row_of_distances.append(index_distance[1])
            neighbor_indices.append(row_of_neighbor_indices)
            distances.append(row_of_distances)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Check if X contains categorical attributes
        if type(X_test[0][0]) == str:
            categorical = True
        else:
            categorical = False

        # Like the kneighbors function, need the neighbor indices to gather the corresponding
        # values in y_train, then use majority rule to predict a classification for X_test
        # Get indices and distances
        row_indexes_dists = classifiers_utils.get_indices_and_dists(self, X_test, categorical)
        # Get the k_closest neighbors indexes and values for each
        # row within total distances
        total_neighbors = classifiers_utils.get_kclosest_neighbors(self, row_indexes_dists)
        # Get the indexes of the k_closest_neighbors
        neighbor_indices = []
        for row in total_neighbors:
            row_of_neighbor_indices = []
            for index_distance in row:
                row_of_neighbor_indices.append(index_distance[0])
            neighbor_indices.append(row_of_neighbor_indices)
        # Find the corresponding classification in y_train
        k_neighbors_classifications = []
        for row in neighbor_indices:
            k_neighbors_classification = []
            for index in row:
                k_neighbors_classification.append(self.y_train[index])
            k_neighbors_classifications.append(k_neighbors_classification)
        # Use majority rule to predict the class
        y_predicted = classifiers_utils.find_majority_2D(k_neighbors_classifications)

        return y_predicted


##############################################################
# NAIVE BAYES CLASSIFIER
##############################################################
class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(dict): The prior probabilities computed for each
            label in the training set.
        posteriors(dict of dict): The posterior probabilities computed for each
            attribute value/label pair in the training set.
        classifications(list): The classifications present in X_train that have
            been fit to the classifier
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.classifications = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """

        X_copy = X_train.copy()
        y_copy = y_train.copy()

        # Get unique class labels
        classifications_count, classifications = classifiers_utils.group_by_classification(y_train)
        self.classifications = classifications

        # Compute priors
        priors = {}
        for classification in classifications:
            priors[classification] = 0
            for idx in range(len(y_copy)):
                if y_copy[idx] == classification:
                    priors[classification] += 1
            priors[classification] = priors[classification] / len(y_copy)
        # Compute posteriors
        posteriors = classifiers_utils.compute_posteriors(X_copy, y_copy, classifications, classifications_count)

        self.priors = priors
        self.posteriors = posteriors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        predictions = {}
        for instance in X_test:
            for classification in self.classifications:
                prediction = 1
                for attribute_idx , value in enumerate(instance):
                    keys_list = self.posteriors[classification][attribute_idx].keys()
                    for key in keys_list:
                        if value == key:
                            prediction *= self.posteriors[classification][attribute_idx][key]
                prediction *= self.priors[classification]
                predictions[classification] = prediction
            # Choose the correct instance
            prediction = classifiers_utils.find_naive_prediction(predictions, self.classifications, self.priors)
            y_predicted.append(prediction)

        return y_predicted