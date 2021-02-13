================
OtolReader v1.0
================

Introduction:
-------------
OtolReader supports automated classification of hatchery marks in salmon otoliths. The package provides a Jupyter Notebook-based utility that allows a human to easily extract marked subsamples from an otolith image. These samples can be stored in a local directory. The samples can then be used to train a machine learning algorithm for otolith classification. The software automatically processes the samples to generate a list of 151 numbers summarizing each sample. The set of all sample summary vectors are then used to train a neural network to implement the classification.


File structure:
---------------

OtolithAnalysis
+++++++++++++++
A package that stores otolith processing utilities.

feature_functions
*****************
    stores several otolith-specific image processing functions that can be used by higher-level algorithms.

draw_samples_2
    Used to draw rectangles at predetermined locations in an image, based on x, y, and rotation coordinates. Useful for visualizing sample selections.
draw_samples_3
    Similar functionality to draw_samples_2, but a more efficient implementation. Used to draw rectangles at predetermined locations in an image, based on x, y, and rotation coordinates. Useful for visualizing sample selections.

extract_samples_2
    Extracts samples from an image given coordinates and rotation. Assumes that the whole image is rotated before applying the coordinates.

extract_samples_3
    Extracts samples from an image given coordinates and rotation. Assumes that the coordinates are applied, then the sample is extracted, then rotated and cropped to the final size.

fcn_mark_im_ind
    Computes the mark index and image index (starting from 0) based on the raw image index and the number of images per mark. For example, If the raw image index is 5 and their are two images per mark, then the function would return mark index 2 and image index 1.

high_low
    computes upper, lower, left and right indexes to use in extracting a sub-array, given the center coordinates and target width and height. Accounts for cases where the target sample is reduced in size due to overlapping the boundary of the original array.

cross_validation
    This function implements ten-fold cross validation on a data set. It is rarely used now because it is not designed for work with AWS, but can reduce the overhead in implementing cross-validation tests.

fft
***
defines two functions that implement a fast fourrier transform algorithm and use it to interpret otolith image samples.

fft_score
    compresses an input 2D sample to a 1D array, computes a second derivative of the 1D array, calculates the FFT of the result and returns the sum of the f_fft over the range of frequencies specified by the user (defaults to 9:12)

fft_finder
    Iterates across an image and returns the indexes in the image with the highest fft score (as calculated by the fft_score function)

Directory containing samples extracted from the five original image classes (3,5H10, 1,6H, None, 4n,2n,2H, 6,2H).

TrainingDatabaseMaker.ipynb
+++++++++++++++++++++++++++++
A jupyter notebook that can be used to efficiently extract marked samples from otolith images.

InternalTransferNetworkScripts
+++++++++++++++++++++++++++++++
A directory of modules for creating and using classification neural networks.

1DTrainingArrayMaker.py
    A script used to generate a training array. The script process samples from SampleDatabase to create to generate synthetic samples, process them into 151 element vectors, and store all of them in a single array.

ClassificationNetworkMarkScoreTable.py
    A script that iterates through models generated for cross validation and calculates the accuracy of
    classifications for each fold.

TwoNetworkTesting.py
    A script used to train a selection and classicification array for cross validation.