from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import cv2 as cv
from joblib import Parallel, delayed
import multiprocessing
import imutils

sys.path.append('..')
import OtolithAnalysis as oa
from OtolithAnalysis.feature_functions import high_low
import time

# ----------------------------------------------Constants--------------------------
mark_list = ['3,5H10', '1,6H', 'none', '6,2H', '4n,2n,2H']
n_img = 30  # Number of images per class
im_per_fold = 15 # Number of images in each fold
n_fold = 10
imdir = "../../OtolithImages/"
fname_base = 'ClassifierNetworks/classnet_'
# selector_str = 'model_augmentedTraining_'
shape = [800, 240]
model_in_len = 151
ang = np.linspace(-90, 90, 9)
x_step = 120

# ----------------------------------------Analysis------------------------
with open('crossval_order.p', 'rb') as f:
    imorder = pickle.load(f)


def fcn_mark_im_ind(im_ind_raw):
    """
    Calculates the mark an image index based on the total image index
    For example, im_ind_raw is 52 and n_img is 30, then mark_ind is 1
    and im_ind is 22
    """
    mark_ind = int(im_ind_raw / n_img)
    im_ind = im_ind_raw - mark_ind * n_img
    return mark_ind, im_ind


def find_samples(img_in, score_fcn, percentile,
                 model,
                 angles=[0],
                 s_e=np.array([800, 240]),
                 x_step=120, y_step=120,
                 print_status=False):
    """
    Locate samples in an image that are likely to contain a mark according to score_fcn
    :param im: a numpy array
    :param score_fcn:
    """
    score_table = []
    diag = int(np.sqrt(shape[0] ** 2 + shape[1] ** 2) / 2) + 1
    nsteps = int((img_in.shape[1] - 2 * diag) / x_step) + 1
    counter = 1
    for x in range(diag, img_in.shape[1] - diag, x_step):
        if print_status:
            print("Evaluating step " + str(counter) + " of " + str(nsteps))
        counter += 1
        for y in range(diag, img_in.shape[0] - diag, y_step):
            xlow, xhigh, ylow, yhigh = high_low(x, y, diag, diag, img_in.shape)
            for ang_ind in range(len(angles)):
                img2 = imutils.rotate(img_in[ylow:yhigh, xlow:xhigh], angles[ang_ind])
                xl, xh, yl, yh = high_low(int(img2.shape[1] / 2), int(img2.shape[0] / 2),
                                          int(shape[1] / 2), int(shape[0] / 2), img2.shape)
                score = score_fcn(img2[yl:yh, xl:xh], model)
                score_table.append([x, y, angles[ang_ind], score])
    score_table = np.array(score_table)
    cond = (score_table[:, 3] >=
            np.percentile(score_table[:, 3], percentile))
    maxscores = score_table[cond, :]
    return maxscores


def net_score_fcn(samp_array, model_in):
    smd2ydx2 = oa.finder_network.fcn_smoothed_d2ydx2(samp_array)
    return mark_prob(smd2ydx2, model_in)


def mark_prob(smd2ydx2, model_in):
    return 1 - model_in.predict(np.expand_dims(np.expand_dims(smd2ydx2, axis=0), axis=2))[0, 2]


def find_marks(fold_ind):
    save_scores = []
    st_save = []
    smoothed_samples = []
    print("Currently evaluating fold: {:0.0f}".format(fold_ind))
    fname = fname_base + str(fold_ind) + '.h5'
    model = keras.models.load_model(fname, compile=False)

    for im_ind in range(0, 50):
        # iterate over all images and identify likely mark samples:
        print('fold: {:0.0f}, image: {:0.0f}'.format(fold_ind, im_ind), flush=True)
        im_lab = imorder[im_ind]
        mark_ind, im_file_ind = fcn_mark_im_ind(im_lab)
        fname = '../../OtolithImages/' + mark_list[mark_ind] + '/' + str(im_file_ind) + '.jpg'
        img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
        st = find_samples(img, net_score_fcn, 99.9, model, angles=ang)
        st_save.append(st)
        samps = oa.feature_functions.extract_samples_3(img, st, [800, 240])
        d2ydx2 = np.zeros([len(samps), model_in_len])
        for s_ind in range(len(samps)):
            d2ydx2[s_ind, :] = oa.finder_network.fcn_smoothed_d2ydx2(samps[s_ind])
        smoothed_samples.append(d2ydx2)
        scores = model.predict(np.expand_dims(d2ydx2, axis=2))
        save_scores.append(scores)

    with open('2019_08_18_ClassifierNetworkForSelection_ScoreTable_' + str(fold_ind) + '.p', 'wb') as f:
        pickle.dump(st_save, f)

    with open('2019_08_18_ClassifierNetworkForSelection_SmoothedSamples_' + str(fold_ind) + '.p', 'wb') as f:
        pickle.dump(smoothed_samples, f)

    with open('2019_08_18_ClassifierNetworkForSelection_Scores_' + str(fold_ind) + '.p', 'wb') as f:
        pickle.dump(save_scores, f)


with multiprocessing.Pool() as pool:
    pool.map(find_marks, range(n_fold))
