############################################################################
# Name: Liam Navarre, Connor Deide
# Class: CPSC 322, Fall 2022
# Final Project
# 12/14/2022
#
# Description: This file contains unit tests for the classifiers in 
# classifiers.py. Unit tests verify the functionality of the fit and predict
# functions for each classifier.
############################################################################

import numpy as np

from classifier_models import evaluators
from classifier_models.classifiers import (
    MyDummyClassifier,
    MyNaiveBayesClassifier,
    MyDecisionTreeClassifier,
    )


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


############################################################################
# Test Decision Tree Classifier
############################################################################
def test_decision_tree_classifier_fit():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
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
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", \
        "False", "True", "True", "True", "True", "True", "False"]

    # Fit the classifier
    tree_clf1 = MyDecisionTreeClassifier()
    tree_clf1.fit(X_train_interview, y_train_interview)

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    # Expected returns
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Junior", 
                    ["Attribute", "att3",
                        ["Value", "no", 
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes", 
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ]
            ]

    # Assert
    assert tree_clf1.tree == tree_interview 

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
    tree_clf2 = MyDecisionTreeClassifier()
    tree_clf2.fit(X_train_iphone, y_train_iphone)

    # Desk check for tree created in fit
    tree_iphone = \
        ["Attribute", "att0",
            ["Value", 1,
                ["Attribute", "att1",
                    ["Value", 3,
                        ["Leaf", "no", 2, 5]
                    ],
                    ["Value", 2,
                        ["Attribute", "att2",
                            ["Value", "fair",
                                ["Leaf", "no", 1, 2]
                            ],
                            ["Value", "excellent",
                                ["Leaf", "yes", 1, 2]
                            ]
                        ]
                    ],
                    ["Value", 1,
                        ["Leaf", "yes", 1, 5]
                    ]
                ]
            ],
            ["Value", 2,
                ["Attribute", "att2",
                    ["Value", "fair",
                        ["Leaf", "yes", 6, 10]
                    ],
                    ["Value", "excellent",
                        ["Attribute", "att1",
                            ["Value", 2,
                                ["Leaf", "no", 2, 4 ]
                            ],
                            ["Value", 1,
                                ["Leaf", "no", 2, 4]
                            ]
                        ]
                    ]
                ]
            ]
        ]

    # Assert
    assert tree_clf2.tree == tree_iphone

def test_decision_tree_classifier_predict():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
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
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", \
        "False", "True", "True", "True", "True", "True", "False"]

    # Fit the classifier
    tree_clf1 = MyDecisionTreeClassifier()
    tree_clf1.fit(X_train_interview, y_train_interview)
    # Predict
    X_test_clf1 = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    predicted_tree_clf1 = tree_clf1.predict(X_test_clf1)

    # Create expected returns
    expected_tree_clf1 = ["True", "False"]

    # Assert
    assert predicted_tree_clf1 == expected_tree_clf1

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
    tree_clf2 = MyDecisionTreeClassifier()
    tree_clf2.fit(X_train_iphone, y_train_iphone)
    # Predict
    X_test_clf2 = [[2, 2, "fair"], [1, 1, "excellent"]]
    predicted_tree_clf2 = tree_clf2.predict(X_test_clf2)

    # Create expected returns
    expected_tree_clf2 = ["yes", "yes"]

    # Assert
    assert predicted_tree_clf2 == expected_tree_clf2