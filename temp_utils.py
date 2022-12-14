import os
import math
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project_application import project_utils
importlib.reload(project_utils)
from classifier_models import classifiers
importlib.reload(classifiers)
from classifier_models import classifier_utils
importlib.reload(classifier_utils)
from classifier_models import evaluators as eval
importlib.reload(eval)
from classifier_models.classifiers import MyDummyClassifier as Dummy_clf
from classifier_models.classifiers import MyNaiveBayesClassifier as NaiveBayes_clf


def about_descretizer(about_ser):
    character_list = []
    return_list = []
    for about_section in about_ser:
        if "Dwight" in about_section:
            character_list.append("Dwight")
        if "Michael" in about_section:
            character_list.append("Michael")
        if "Jim" in about_section:
            character_list.append("Jim")
        if "Pam" in about_section:
            character_list.append("Pam")
        if "Jan" in about_section:
            character_list.append("Jan")
        if "Andy" in about_section:
            character_list.append("Andy")
        if "Kevin" in about_section:
            character_list.append("Kevin")
        if "Angela" in about_section:
            character_list.append("Angela")
        if "Toby" in about_section:
            character_list.append("Toby")
        if "Holly" in about_section:
            character_list.append("Holly")
        if "Kelly" in about_section:
            character_list.append("Kelly")
        if "Stanley" in about_section:
            character_list.append("Stanley")
        if "Ryan" in about_section:
            character_list.append("Ryan")
        if "Meredith" in about_section:
            character_list.append("Meredith")
        if "Karen" in about_section:
            character_list.append("Karen")
        if "Darryl" in about_section:
            character_list.append("Darryl")
        if "Gabe" in about_section:
            character_list.append("Gabe")
        if "Roy" in about_section:
            character_list.append("Roy")
        if "Oscar" in about_section:
            character_list.append("Oscar")
        if "Phyllis" in about_section:
            character_list.append("Phyllis")
        if "Creed" in about_section:
            character_list.append("Creed")
        else:
            character_list.append("Other")
        return_list.append(character_list)
        character_list = []
    return return_list