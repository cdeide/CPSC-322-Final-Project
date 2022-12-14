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

from classifier_models import classifier_utils

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
        self.most_common_label = classifier_utils.find_majority_1D(y_train)

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
        classifications_count, classifications = classifier_utils.group_by_classification(y_train)
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
        posteriors = classifier_utils.compute_posteriors(X_copy, y_copy, classifications, classifications_count)

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
            prediction = classifier_utils.find_naive_prediction(predictions, self.classifications, self.priors)
            y_predicted.append(prediction)

        return y_predicted


##############################################################
# Decision Tree Classifier
##############################################################
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.header = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """

        # Build string of arbitrary attribute names
        available_attributes = []
        for i in range(len(X_train[0])):
            available_attributes.append("att" + str(i))
        self.header = available_attributes.copy()
        # Build dictionary of attribute domains
        attribute_domains = {}
        for i in range(len(X_train[0])):
            attribute_domains["att" + str(i)] = []
            for row in X_train:
                if row[i] not in attribute_domains["att" + str(i)]:
                    attribute_domains["att" + str(i)].append(row[i])

        # Combine X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]

        # Call TDIDT
        tree = classifier_utils.tdidt(self, train, available_attributes, attribute_domains)
        self.tree = tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            prediction = classifier_utils.find_tree_prediction(self.tree, instance)
            # print("Instance:", instance)
            # print("Prediction:", prediction)
            predictions.append(prediction)
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # Call recursive function
        classifier_utils.find_rule(self, self.tree, "", attribute_names, class_name)

##############################################################
# Random Forest Classifier
##############################################################
class MyRandomForestClassifier:
    """
    Represents a Random Forest Classifier

    Attributes:
        remainder_set(list of list): The 2D list of training data after creating the random stratified
            test set from a third of the original dataset
        N (int): The number of total decision trees to make.
        M (int): The number of most accurate decision trees to create rand_forest.
        F (int): The number of attributes to consider splitting on for each node in a decision tree
        rand_forest(list of MyDecisionTreeClassifiers): The decision tree classifiers making up the
            random forest
        rand_state (int): Value to seed the random class
    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def __init__(self, N, M, F, rand_state=None):
        """
        Class initializer
        """
        self.remainder_set = None
        self.N = N
        self.M = M
        self.F = F
        self.rand_forest = None
        self.rand_state = rand_state

    def fit(self, remainder_set):
        """
        Fits a random forest classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            remainder_set (list of list): 2D list of data to fit the classifier to
        """

        self.remainder_set = remainder_set
        # Build random forest classifier from remainder_set and M, N, and F values
        classifier_utils.get_rand_forest(self, remainder_set)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for row in X_test:
            y_predicted.append(classifier_utils.get_forest_prediction(self.rand_forest, row))
        return y_predicted

