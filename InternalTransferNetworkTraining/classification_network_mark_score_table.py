from __future__ import absolute_import, division, print_function
from tensorflow import keras
import numpy as np
import os
import pickle
import cv2 as cv
import imutils
from otolreader import OtolithAnalysis as oa
from otolreader.OtolithAnalysis.feature_functions import high_low


def fcn_mark_im_ind(im_ind_raw, n_img=30):
    """
    Calculates the mark an image index based on the total image index
    For example, im_ind_raw is 52 and n_img is 30, then mark_ind is 1
    and im_ind is 22
    :param im_ind_raw: the raw image number across all marks
    :param n_img: the number of images per mark
    :return mark_ind: the mark index of the image (from the range 0:n_marks)
    :return im_ind: the mark-specific index of the image (from the range 0:n_img)
    """
    mark_ind = int(im_ind_raw / n_img)
    im_ind = im_ind_raw - mark_ind * n_img
    return mark_ind, im_ind


def find_samples(img_in, score_fcn, percentile, model, angles=[0], shape=[800, 240], x_step=120, y_step=120,
                 print_status=False):
    """
    Locate samples in an image that are likely to contain a mark according to score_fcn
    :param img_in: a numpy array containing pixel intensitities
    :param score_fcn: a function that returns larger numbers for samples more likely to contain a mark
    :param percentile: the threshold percentile for samples to store
    :param model: the model to be used as part of score_fcn
    :param angles: a list of rotation angles to consider
    :param shape: the shape of samples to extract
    :param x_step: the number of pixels to move along x between steps
    :param y_step: the number of pixels to move along y between steps
    :param print_status: If True, then print status updates while the function completes.
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
    """
    A function to claculate the probability that a sample contains a mark
    """
    smd2ydx2 = oa.finder_network.fcn_smoothed_d2ydx2(samp_array)
    return mark_prob(smd2ydx2, model_in)


def mark_prob(smd2ydx2, model_in):
    """
    Uses model_in (a classification network) to estimate the probability that a 1D sample contains a mark
    """
    return 1 - model_in.predict(np.expand_dims(np.expand_dims(smd2ydx2, axis=0), axis=2))[0, 2]


def find_marks(fname_base, fold_ind, imorder, im_dir, mark_list, ang, model_in_len, max_im_ind, dirout):
    """
    Iterates through all images and provides mark-unmarked scores and locations. Also extracts 1D reductions of
    samples likely to contain marks. Saves the score table, scores and samples to files in dir_out.
    :param fname_base: the base name for models to load, excluding the fold index
    :param fold_ind: the fold index associated with the model to load
    :param imorder: the cross validation image order (an array)
    :param im_dir: directory of images to load (should have subdirectories for each
    :param mark_list: the list of marks, in order (eg. ['3,5H10', 'None', '6,2H'])
    :param ang: the list of rotation angles to consider
    :param model_in_len: the length of 1D arrays used to train the classification array
    """
    save_scores = []
    st_save = []
    smoothed_samples = []
    print("Currently evaluating fold: {:0.0f}".format(fold_ind))
    fname = fname_base + str(fold_ind) + '.h5'
    model = keras.models.load_model(fname, compile=False)

    for im_ind in range(0, max_im_ind):
        # iterate over all images and identify likely mark samples:
        print('fold: {:0.0f}, image: {:0.0f}'.format(fold_ind, im_ind), flush=True)
        im_lab = imorder[im_ind]
        mark_ind, im_file_ind = fcn_mark_im_ind(im_lab)
        fname = os.path.join(im_dir, mark_list[mark_ind], str(im_file_ind) + '.jpg')
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
    fpath = os.path.join(dirout, 'ClassifierNetworkForSelection_ScoreTable_' + str(fold_ind) + '.p')
    with open(fpath, 'wb') as f:
        pickle.dump(st_save, f)
    fpath = os.path.join(dirout, 'ClassifierNetworkForSelection_SmoothedSamples_' + str(fold_ind) + '.p')
    with open(fpath, 'wb') as f:
        pickle.dump(smoothed_samples, f)
    fpath = os.path.join(dirout, 'ClassifierNetworkForSelection_Scores_' + str(fold_ind) + '.p')
    with open(fpath, 'wb') as f:
        pickle.dump(save_scores, f)
