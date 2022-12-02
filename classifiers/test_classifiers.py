############################################################################
# Name: Liam Navarre, Connor Deide
# Class: CPSC 322, Fall 2022
# Final Project
# 12/14/2022
#
# Description: This file contains unit tests for the classifiers in 
# classifiers.py. Unit tests verify the functionality of the fit and predict
# functions for each classifier. (As well as the kneighbors funtionality in
# the KNeighbors Classifier)
############################################################################

import numpy as np

from classifiers import MyNaiveBayesClassifier
from classifiers import MyKNeighborsClassifier
from classifiers import MyDummyClassifier


############################################################################
# Test Dummy Classifier
############################################################################
def test_dummy_classifier_fit():
    # Create data for test case A
    np.random.seed(0)
    X_train_A = [[value] for value in range(100)]
    y_train_A = list(np.random.choice(["yes", "no"], 100, replace=True, p = [0.7, 0.3]))
    lin_clf_A = MyDummyClassifier()
    lin_clf_A.fit(X_train_A, y_train_A)

    # Test
    # Desk Check
    most_common_solution_A = "yes"
    assert lin_clf_A.most_common_label == most_common_solution_A

    # Create data for test case B
    X_train_B = [[value] for value in range(100)]
    y_train_B = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, \
        p = [0.2, 0.6, 0.2]))
    lin_clf_B = MyDummyClassifier()
    lin_clf_B.fit(X_train_B, y_train_B)

    # Test
    # Desk Check
    most_common_solution_B = "no"
    assert lin_clf_B.most_common_label == most_common_solution_B

    # Create data for test case C
    X_train_C = [[value] for value in range(100)]
    y_train_C = list(np.random.choice(["dog", "cat", "parrot", "whale"], 100, replace=True, \
        p = [0.2, 0.1, 0.3, 0.4]))
    lin_clf_C = MyDummyClassifier()
    lin_clf_C.fit(X_train_C, y_train_C)

    # Test
    # Desk Check
    most_common_solution_C = "whale"
    assert lin_clf_C.most_common_label == most_common_solution_C

def test_dummy_classifier_predict():
    # Create data for test case A
    np.random.seed(0)
    X_train_A = [[value] for value in range(100)]
    y_train_A = list(np.random.choice(["yes", "no"], 100, replace=True, p = [0.7, 0.3]))
    X_test_A = []
    for _ in range(3):
        X_test_A.append(np.random.randint(100))
    dum_clf_A = MyDummyClassifier()
    dum_clf_A.fit(X_train_A, y_train_A)

    # Test
    # Desk Check
    most_common_solutions_A = ["yes", "yes", "yes"]
    assert dum_clf_A.predict(X_test_A) == most_common_solutions_A

    # Create data for test case B
    X_train_B = [[value] for value in range(100)]
    y_train_B = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, \
        p = [0.2, 0.6, 0.2]))
    X_test_B = []
    for _ in range(2):
        X_test_B.append(np.random.randint(100))
    dum_clf_B = MyDummyClassifier()
    dum_clf_B.fit(X_train_B, y_train_B)

    # Test
    # Desk Check
    most_common_solutions_B = ["no", "no"]
    assert dum_clf_B.predict(X_test_B) == most_common_solutions_B 

    # Create data for test case C
    X_train_C = [[value] for value in range(100)]
    y_train_C = list(np.random.choice(["dog", "cat", "parrot", "whale"], 100, replace=True, \
        p = [0.2, 0.1, 0.3, 0.4]))
    X_test_C = []
    for _ in range(3):
        X_test_C.append(np.random.randint(100))
    dum_clf_C = MyDummyClassifier()
    dum_clf_C.fit(X_train_C, y_train_C)

    # Test
    # Desk Check
    most_common_solutions_C = ["whale", "whale", "whale"]
    assert dum_clf_C.predict(X_test_C) == most_common_solutions_C


############################################################################
# Test KNeighbors Classifier
############################################################################
def test_kneighbors_classifier_kneighbors():
    # In-class training set 1 (4 instances)
    X_train_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_example1 = ["bad", "bad", "good", "good"]
    X_test_example1 = [[0.33, 1]]
    knn_clf_example1 = MyKNeighborsClassifier(3)
    knn_clf_example1.fit(X_train_example1, y_train_example1)
    # Create expected returns
    example1_distances_expected = [[0.670, 1.00, 1.053]]
    example1_neighbor_indices_expected = [[0, 2, 3]]
    # Get actual returns
    example1_distances_returned, example1_neighbor_indices_returned = \
        knn_clf_example1.kneighbors(X_test_example1)
    # Assert
    assert example1_neighbor_indices_returned == example1_neighbor_indices_expected
    assert np.allclose(example1_distances_returned, example1_distances_expected, 0.01)

    # In-class training set 2 (8 instances)
    X_train_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_example2 = [[2, 3]]
    knn_clf_example2 = MyKNeighborsClassifier(3)
    knn_clf_example2.fit(X_train_example2, y_train_example2)
    # Create Expected returns
    example2_distances_expected = [[1.414, 1.414, 2.0]]
    example2_neighbor_indices_expected = [[0, 4, 6]]
    # Get actual returns
    example2_distances_returned, example2_neighbor_indices_returned = \
        knn_clf_example2.kneighbors(X_test_example2)
    # Assert
    assert example2_neighbor_indices_returned == example2_neighbor_indices_expected
    assert np.allclose(example2_distances_returned, example2_distances_expected, 0.01)

    # Bramer training set
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    X_test_bramer_example = [[9.1, 11.0]]
    knn_clf_bramer_example = MyKNeighborsClassifier(5)
    knn_clf_bramer_example.fit(X_train_bramer_example, y_train_bramer_example)
    # Create expected returns
    bramer_example_distances_expected = [[0.608, 1.237, 2.202, 2.802, 2.915]]
    bramer_example_neighbor_indices_expected = [[6, 5, 7, 4, 8]]
    # Get actual returns
    bramer_example_distances_returned, bramer_example_neighbor_indices_returned = \
        knn_clf_bramer_example.kneighbors(X_test_bramer_example)
    # Assert
    assert bramer_example_neighbor_indices_returned == bramer_example_neighbor_indices_expected
    assert np.allclose(bramer_example_distances_returned, bramer_example_distances_expected, 0.01)

def test_kneighbors_classifier_predict():
    # In-class training set 1 (4 instances)
    X_train_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_example1 = ["bad", "bad", "good", "good"]
    X_test_example1 = [0.33, 1]
    knn_clf_example1 = MyKNeighborsClassifier(3)
    knn_clf_example1.fit(X_train_example1, y_train_example1)
    # Create Expected returns
    example1_y_predicted_solution = ["good"]
    # Get actual returns
    example1_y_predicted = knn_clf_example1.predict([X_test_example1])
    # Assert
    assert example1_y_predicted == example1_y_predicted_solution

    # In-class training set 2 (8 instances)
    X_train_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_example2 = [2, 3]
    knn_clf_example2 = MyKNeighborsClassifier(3)
    knn_clf_example2.fit(X_train_example2, y_train_example2)
    # Create Expected returns
    example2_y_predicted_solution = ["yes"]
    # Get actual returns
    example2_y_predicted = knn_clf_example2.predict([X_test_example2])
    # Assert
    assert example2_y_predicted == example2_y_predicted_solution

    # Bramer training set
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    X_test_bramer_example = [9.1, 11.0]
    knn_clf_bramer_example = MyKNeighborsClassifier(5)
    knn_clf_bramer_example.fit(X_train_bramer_example, y_train_bramer_example)
    # Create expected returns
    bramer_example_y_predicted_solution = ["+"]
    # Get actual returns
    bramer_example_y_predicted = knn_clf_bramer_example.predict([X_test_bramer_example])
    # Assert
    assert bramer_example_y_predicted == bramer_example_y_predicted_solution


############################################################################
# Test Naive Bayes Classifier
############################################################################
def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    # Fit the classifier
    naive_bayes_clf1 = MyNaiveBayesClassifier()
    naive_bayes_clf1.fit(X_train_inclass_example, y_train_inclass_example)

    # Create expected returns
    expected_priors_clf1 = {"yes": 5/8,
                            "no": 3/8}
    expected_posteriors_clf1 = {"yes":
                                    {0:
                                        {1: 4/5,
                                        2: 1/5},
                                    1:
                                        {5: 2/5,
                                        6: 3/5}},
                                "no":
                                    {0:
                                        {1: 2/3,
                                        2: 1/3},
                                    1:
                                        {5: 2/3,
                                        6: 1/3}}}

    # Assert
    assert naive_bayes_clf1.priors == expected_priors_clf1
    assert naive_bayes_clf1.posteriors == expected_posteriors_clf1

    # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", \
        "yes", "yes", "yes", "yes", "no", "yes"]

    # Fit the classifier
    naive_bayes_clf2 = MyNaiveBayesClassifier()
    naive_bayes_clf2.fit(X_train_iphone, y_train_iphone)

    # Create expected returns
    expected_priors_clf2 = {"yes": 10/15,
                            "no": 5/15}
    expected_posteriors_clf2 = {"yes":
                                    {0:
                                        {1: 2/10,
                                        2: 8/10},
                                    1:
                                        {3: 3/10,
                                        2: 4/10,
                                        1: 3/10},
                                    2:
                                        {"fair": 7/10,
                                        "excellent": 3/10}},
                                "no": 
                                    {0:
                                        {1: 3/5,
                                        2: 2/5},
                                    1:
                                        {3: 2/5,
                                        2: 2/5,
                                        1: 1/5,},
                                    2:
                                        {"fair": 2/5,
                                        "excellent": 3/5}}}

    # Assert
    assert naive_bayes_clf2.priors == expected_priors_clf2
    assert naive_bayes_clf2.posteriors == expected_posteriors_clf2

    # Bramer 3.2 train dataset
    header_bramer = ["day", "season", "wind", "rain", "class"]
    X_train_bramer = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_bramer = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]

    # Fit the classifier
    naive_bayes_clf3 = MyNaiveBayesClassifier()
    naive_bayes_clf3.fit(X_train_bramer, y_train_bramer)

    # Create expected returns
    expected_priors_clf3 = {"on time": 14/20,
                        "late": 2/20,
                        "very late": 3/20,
                        "cancelled": 1/20}
    expected_posteriors_clf3 = {"on time":
                                    {0: 
                                        {"weekday": 9/14,
                                        "saturday": 2/14,
                                        "holiday": 2/14,
                                        "sunday": 1/14},
                                    1:
                                        {"spring": 4/14,
                                        "winter": 2/14,
                                        "summer": 6/14,
                                        "autumn": 2/14},
                                    2:
                                        {"none": 5/14,
                                        "high": 4/14,
                                        "normal": 5/14},
                                    3:
                                        {"none": 5/14,
                                        "slight": 8/14,
                                        "heavy": 1/14}},
                                "late":
                                    {0:
                                        {"weekday": 1/2,
                                        "saturday": 1/2,
                                        "holiday": 0/2,
                                        "sunday": 0/2},
                                    1:
                                        {"spring": 0/2,
                                        "winter": 2/2,
                                        "summer": 0/2,
                                        "autumn": 0/2},
                                    2:
                                        {"none": 0/2,
                                        "high": 1/2,
                                        "normal": 1/2},
                                    3:
                                        {"none": 1/2,
                                        "slight": 0/2,
                                        "heavy": 1/2}},
                                "very late":
                                    {0: 
                                        {"weekday": 3/3,
                                        "saturday": 0/3,
                                        "holiday": 0/3,
                                        "sunday": 0/3},
                                    1:
                                        {"spring": 0/3,
                                        "winter": 2/3,
                                        "summer": 0/3,
                                        "autumn": 1/3},
                                    2:
                                        {"none": 0/3,
                                        "high": 1/3,
                                        "normal": 2/3},
                                    3:
                                        {"none": 1/3,
                                        "slight": 0/3,
                                        "heavy": 2/3}},
                                "cancelled":
                                    {0:
                                        {"weekday": 0/1,
                                        "saturday": 1/1,
                                        "holiday": 0/1,
                                        "sunday": 0/1},
                                    1:
                                        {"spring": 1/1,
                                        "winter": 0/1,
                                        "summer": 0/1,
                                        "autumn": 0/1},
                                    2:
                                        {"none": 0/1,
                                        "high": 1/1,
                                        "normal": 0/1},
                                    3:
                                        {"none": 0/1,
                                        "slight": 0/1,
                                        "heavy": 1/1}
                                    }
                                }

    # Assert
    assert naive_bayes_clf3.priors == expected_priors_clf3
    assert naive_bayes_clf3.posteriors == expected_posteriors_clf3

def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_inclass_example = [[1, 5]]

    # Fit the classifier
    naive_bayes_clf1 = MyNaiveBayesClassifier()
    naive_bayes_clf1.fit(X_train_inclass_example, y_train_inclass_example)
    # Predict
    y_predicted_clf1 = naive_bayes_clf1.predict(X_test_inclass_example)

    # Create expected returns
    y_predicted_clf1_solution = ["yes"]

    # Assert
    assert y_predicted_clf1 == y_predicted_clf1_solution

    # RQ5 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", \
        "yes", "yes", "yes", "yes", "no", "yes"]
    X_test_iphone = [[2, 2, "fair"], [1, 1, "excellent"]]

    # Fit the classifier
    naive_bayes_clf2 = MyNaiveBayesClassifier()
    naive_bayes_clf2.fit(X_train_iphone, y_train_iphone)
    # Predict
    y_predicted_clf2 = naive_bayes_clf2.predict(X_test_iphone)

    # Create expected returns
    y_predicted_clf2_solution = ["yes", "no"]

    # Assert
    assert y_predicted_clf2 == y_predicted_clf2_solution

    # Bramer 3.2 train dataset
    header_bramer = ["day", "season", "wind", "rain", "class"]
    X_train_bramer = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_bramer = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    X_test_bramer = [["weekday", "winter", "high", "heavy"], \
                    ["weekday", "summer", "high", "heavy"], \
                    ["sunday", "summer", "normal", "slight"]]

    # Fit the classifier
    naive_bayes_clf3 = MyNaiveBayesClassifier()
    naive_bayes_clf3.fit(X_train_bramer, y_train_bramer)
    # Predict
    y_predicted_clf3 = naive_bayes_clf3.predict(X_test_bramer)

    # Create expected returns
    y_predicted_clf3_solution = ["very late", "on time", "on time"]

    # Assert
    assert y_predicted_clf3 == y_predicted_clf3_solution