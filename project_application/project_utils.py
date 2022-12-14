############################################################################
# Name: Liam Navarre, Connor Deide
# Class: CPSC 322, Fall 2022
# Final Project
# 12/14/2022
#
# Description: This file contains utility functions called in the project
# Jupyter Notebook to help clean and prepare data.
############################################################################

from classifier_models.classifiers import MyRandomForestClassifier 
import classifier_models.classifiers

def about_discretizer(about_data):
    # TODO: make discretizer
    return None

def get_instances(column_list):
    """
    Function returns a dictionary of each uniqe values frequency within a list
    Args:
        column_list (list): list to find the number of instances for each duplicate
    Returns:
        dictionary: dict of instances as keys and however many times they appear as the value
    """
    instance_dict = {}
    for instance in column_list:
        new_instance = True
        keys_list = instance_dict.keys()
        for _ in keys_list:
            if instance in instance_dict:
                # Increment the instance count
                instance_dict[instance] += 1
                new_instance = False
                break
        if new_instance and (instance != 'N/A' or instance != 'NA'):
            instance_dict[instance] = 1

    return instance_dict

def cross_val_predict(X, y, evaluation, classifier, stratify=False):
    """
    This function is used in the pa6.ipynb file. It performs (stratified) cross
    validation on the given dataset and then uses the results to fit training instances to
    the given classifier and predict classifications for new instances

    Args:
        X (list of list): The dataset that needs to be split into training and
            testing data
        y (list): The classifications corresponding to the dataset
        evaluation (MyEvaluation Obj): a MyEvaluation class reference so the function has
            the scope to call evaluation functions on the dataset
        classifier (obj): A specific classifier object from the myclassifiers file
        stratify (bool): Boolean indicating whether to perform cross validation or
            stratified cross validation
    Returns:
        avg_accuracy: The average accuracy of the classifiers predictions
        avg_error_rate: The error rate of the classifiers predictions
        y_true: The correct classifications the classifier is attempting to predict
        y_pred: The predicted classifications
    """

    if stratify:
        folds = evaluation.stratified_kfold_split(X, y, 10)
    else:
        folds = evaluation.kfold_split(X, 10)

    X_train = []
    y_train = []
    X_test = []
    y_true = []

    # "Unfold" the folds
    avg_accuracy = 0
    for fold in folds:
        for train in fold[0]:
            X_train.append(X[train])
            y_train.append(y[train])
        for test in fold[1]:
            X_test.append(X[test])
            y_true.append(y[test])

        classifier.fit(X_train, y_train)
        #print(classifier.tree)

        y_pred = classifier.predict(X_test)
        # print("RECIEVED PRED:", y_pred)
        # print("Y TRUE", y_true)

        # Compare y_test to y_pred for accuracy
        accuracy = evaluation.accuracy_score(y_true, y_pred)
        #print("ACCURACY:", accuracy)
        avg_accuracy += accuracy

    # Get averages
    avg_accuracy = round((avg_accuracy / 10), 2)
    #print("AVG ACCURACY:", avg_accuracy)
    avg_error = round(1.0 - avg_accuracy, 2)

    if stratify:
        return avg_accuracy, avg_error, y_true, y_pred
    else:
        return avg_accuracy, avg_error

def train_test_predict(X, y, evaluation, classifier):

    # Get training and testing data from train_test_split
    X_train, X_test, y_train, y_test = evaluation.train_test_split(\
        X, y, 0.33)
    # Build the remainder set
    remainder_set = []
    for idx in range(len(X_train)):
        remainder = X_train[idx]
        remainder.append(y_train[idx])
        remainder_set.append(remainder)

    # Fit and predict
    classifier.fit(remainder_set)
    y_pred = classifier.predict(X_test)
    # print("RECIEVED PRED:", y_pred)
    # print("Y TRUE", y_test)

    # Compare y_test to y_pred for accuracy
    accuracy = evaluation.accuracy_score(y_test, y_pred)

    # Get averages
    accuracy = round((accuracy / 10), 2)
    error = round(1.0 - accuracy, 2)

    return accuracy, error, y_test, y_pred