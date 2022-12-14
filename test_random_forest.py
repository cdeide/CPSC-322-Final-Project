############################################################################
# Name: Liam Navarre, Connor Deide
# Class: CPSC 322, Fall 2022
# Final Project
# 12/14/2022
#
# Description: This file contains unit tests for the MyRandomForestClassifier in 
# classifiers.py. Unit tests verify the functionality of the fit and predict
# functions.
############################################################################

from classifier_models import evaluators
from classifier_models import classifier_utils
from classifier_models.classifiers import MyRandomForestClassifier


############################################################################
# Test Random Forest Classifier
############################################################################
def test_random_forest_classifier_fit():
    # interview dataset
    X_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_interview = ["False", "False", "True", "True", "True", "False", "True", \
        "False", "True", "True", "True", "True", "True", "False"]

    # Split the data into 1/3 Test set with 2/3 Remainder
    X_train, X_test, y_train, y_test = evaluators.train_test_split(\
        X_interview, y_interview, 0.33)
    # Build remainder from train sets
    remainder_set = []
    for idx in range(len(X_train)):
        remainder = X_train[idx]
        remainder.append(y_train[idx])
        remainder_set.append(remainder)

    rf_clf = MyRandomForestClassifier(20, 5, 2, rand_state=None)
    # Fit the classifier
    rf_clf.fit(remainder_set)

    # Cannot test for random forest correctness as the return changes no matter the random state
    # Will check for correct attributes present in the classifier
    assert rf_clf.N == 20
    assert rf_clf.M == 5
    assert rf_clf.F == 2
    assert rf_clf.remainder_set == remainder_set

def test_random_forest_classifier_predict():
    # interview dataset
    X_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_interview = ["False", "False", "True", "True", "True", "False", "True", \
        "False", "True", "True", "True", "True", "True", "False"]

    # Split the data into 1/3 Test set with 2/3 Remainder
    X_train, X_test, y_train, y_test = evaluators.train_test_split(\
        X_interview, y_interview, 0.33)
    # Build remainder from train sets
    remainder_set = []
    for idx in range(len(X_train)):
        remainder = X_train[idx]
        remainder.append(y_train[idx])
        remainder_set.append(remainder)

    rf_clf = MyRandomForestClassifier(40, 5, 4, rand_state=None)
    # Fit the classifier
    rf_clf.fit(remainder_set)
    # Get predictions from the classifier
    y_predicted = rf_clf.predict(X_test)

    # Cannot test for y_predicted correctness as the return changes no matter the random state
    # Will check for length
    assert len(y_test) == len(y_predicted)

def test_prune_rand_forest():

    rf_clf = MyRandomForestClassifier(20, 3, 2)
    # Make parallel list of trees and accuracies
    rand_forest = [["Tree1"], ["Tree2"], ["Tree3"], ["Tree4"], ["Tree5"]]
    tree_accuracies = [[0.67], [0.53], [0.76], [0.75], [0.80]]
    # Build example random forest
    pruned_rand_forest = classifier_utils.prune_rand_forest(rf_clf, tree_accuracies, rand_forest)

    # Expected pruned forest
    expected_forest = [["Tree5"], ["Tree3"], ["Tree4"]]

    # Assert
    assert len(pruned_rand_forest) == len(expected_forest)
    assert pruned_rand_forest == expected_forest