# CPSC-322-Final-Project

-- Project Overview --
This project uses several Machine Learning Algorithms to produce classification predictions for unseen instances in datasets containing information about TV shows such as 'The Office'. The classification attribute that we are trying to best predict for these datasets is the IMDb rating value. Classifiers used in this project contain a Dummy Classifier, Naive Bayes Classifier, Decision Tree Classifier, and a Random Forest Classifier.

-- How To Run --
Information about our classifications over this dataset can be found in the Jupyter Notebook titled "project_review.ipynb". Running this notebook will produce the information we found important to highlight.

-- Project Organization --
1. Data
  * Datasets in the form of CSV files can be found within the folder titled 'data'
2. Classifiers
  * All files related to the implementation of the classifiers can be found in the folder titled "classifier_models". This includes a file of the classifiers themselves, a classifier utility function file, a file containing evaluators for the classifiers, and a file containing data preperation utilities.
3. Project Application
  * The application of our classifiers over our datasets can be found within the folder titled "project_application". This includes the project_overview Jupyter Notebook, project_utilities to help prepare and classify data within the notebook, and our original project proposal.
4. Unit Tests
  * Unit tests for our classifiers can be found within the root directory of this project. 'test_classifiers.py' contains unit tests for all of our classifiers present except for the random forest classifier. Unit tests for this classifier can be found in the 'test_random_forest.py".
