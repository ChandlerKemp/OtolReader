"""
This module provides a function that classifies otolith images.
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import sys
import cv2 as cv
from . import feature_functions, finder_network
from otolreader.InternalTransferNetworkTraining.classification_network_mark_score_table import net_score_fcn, \
    find_samples


def classify_image(image_path, classification_model, binary_model, binary_threshold,
                   mark_list=['3,5H10', '1,6H', 'none', '6,2H', '4n,2n,2H'],
                   ang=np.linspace(-90, 90, 9),
                   model_in_len=151):
    """
    Classifies an otolith image
    :param image_path: path to an image file
    :param classification_model: a keras model trained for classification
    :param binary_model: a keras model trained for binary distinction between marked and unmarked images
    :param binary_threshold: the threshold of the to designate images as unmarked using the binary array
    :param mark_list: List of possible mark labels (must be the same order used to train the models)
    :param ang: the set of angles to consider when selecting samples
    :param model_in_len: the length of input expected by the models
    """
    st, smoothed_samples, scores = find_marks(image_path, classification_model, ang=ang, model_in_len=model_in_len)
    binary_score = binary_scores(smoothed_samples, binary_model)
    if binary_score[0] > binary_threshold:
        return 'none'
    sctemp = np.average(scores, axis=0)
    yhatclass = np.argmax(sctemp)
    return mark_list[yhatclass]


def find_marks(image_path, classification_model, ang=np.linspace(-90, 90, 9), model_in_len=151, print_status=True):
    """
    Locates likely mark samples within an image. Also extracts 1D reductions of
    samples likely to contain marks. Returns the score table, scores and samples.
    :param image_path: path to an image file
    :param classification_model: a keras model trained for classification
    :param ang: the list of rotation angles to consider
    :param model_in_len: the length of 1D arrays used to train the classification array
    """
    if print_status:
        print("currently evaluating image at " + image_path)
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise TypeError("Image not found at " + image_path)
    st = find_samples(img, net_score_fcn, 99.9, classification_model, angles=ang)
    samps = feature_functions.extract_samples_3(img, st, [800, 240])
    smoothed_samples = np.zeros([len(samps), model_in_len])
    for s_ind in range(len(samps)):
        smoothed_samples[s_ind, :] = finder_network.fcn_smoothed_d2ydx2(samps[s_ind])
    scores = classification_model.predict(np.expand_dims(smoothed_samples, axis=2))
    return st, smoothed_samples, scores


def binary_scores(smoothed_samps, binary_model):
    """
    Provides scores to determine whether an image is marked or unmarked
    :param smoothed_samps: an array of smoothed 1D representations of samples
    :param binary_model: a keras model trained to distinguish between marked and unmarked samples
    """
    score = np.mean(binary_model.predict(np.expand_dims(smoothed_samps, axis=2)), axis=0)
    return score

