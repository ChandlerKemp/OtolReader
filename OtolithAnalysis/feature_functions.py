import numpy as np
import imutils
import cv2 as cv
import copy
import os


def cross_validation(x, y, train_fcn, test_fcn, n_folds=10, train_params={},
                     x_testing=None, test_order=None, return_chosen=False):
    """
    Created 2019-01-29
    Implements cross validation on a training set
    :param x: An array of input feature scores (n_img, n_features)
    :param y: An array of classes (n_img)
    :param train_fcn: A training function
    :param test_fcn: A test function (must accept the first two returns from train_fcn)
    :param n_folds: Number of random training sets to draw
    :param prior: The prior to be used in test_fcn (n_classes)

    :param x_testing: An alternate dataset can be used for testing and training within the cross validation. This
    capacity was developed to accommodate a scenario in which many (~25 up to 150 sub-samples were pulled from images
    for training, classification would be attempted with only about 10 samples per image)
    :return: The average classification rate for each class
    """
    n_imgs = len(x)
    score_temp = np.zeros([n_folds, np.max(y) + 1])
    chosen = []
    all_inds = np.linspace(0, n_imgs - 1, n_imgs, dtype=int)
    if test_order is None:
        test_order = copy.copy(all_inds)
        np.random.shuffle(test_order)
    samps_per_fold = int(len(test_order) / n_folds)
    for counter in range(n_folds):
        test_inds = test_order[counter * samps_per_fold:(counter + 1) * samps_per_fold]
        train_inds = np.setxor1d(all_inds, test_inds)
        chosen.extend(test_inds)
        train_out = train_fcn(x[train_inds], y[train_inds], **train_params)
        if type(train_out) is not tuple:
            train_out = (train_out,)
        ytest = y[test_inds]
        yhat = np.zeros(len(ytest))
        for ind in range(len(test_inds)):
            if x_testing is None:
                yhat[ind] = test_fcn(x[test_inds[ind]], *train_out)
            else:
                yhat[ind] = test_fcn(x_testing[test_inds[ind]], *train_out)
        for ind in range(np.max(y) + 1):
            cond = ytest == ind
            if np.max(cond) > 0:
                score_temp[counter, ind] = np.sum(yhat[cond] == ind)/np.sum(cond)
            else:
                score_temp[counter, ind] = np.nan
    if return_chosen:
        return np.nanmean(score_temp, 0), chosen
    else:
        return np.nanmean(score_temp, 0)


def draw_samples_2(img, scoretable, shape, writerank=False, lineweight=5):
    """
    Created 2019-01-28
    Modified 2019-02-07
    Draws squares at likely mark locations
    :param img: An image from which to draw samples
    :param scoretable: A table of locations, angles, and scores for marks in the image to be extracted
    :param shape: shape of the sample to be extracted
    :param writerank: if True, write the number indicating how high the critscore was compared to other boxes in img
    :param lineweight: weight of lines to draw
    :return samples: a list of samples from the image
    :return img: an image with squares drawn at the saved feature locations
    :return scoretable: A list of the positions, scores, and angles of each square drawn
    """
    img2 = copy.copy(img)
    count = 0
    color = (255, 255, 255)
    for ind in range(len(scoretable)):
        xind = int(scoretable[ind][0])
        yind = int(scoretable[ind][1])
        xlow, xhigh, ylow, yhigh = high_low(xind, yind, int(shape[1]/2), int(shape[0]/2), img.shape)
        img2 = imutils.rotate(img2, scoretable[ind][2])
        img2 = cv.line(img2, (xlow, ylow), (xhigh, ylow), color, lineweight)
        img2 = cv.line(img2, (xlow, ylow), (xlow, yhigh), color, lineweight)
        img2 = cv.line(img2, (xhigh, ylow), (xhigh, yhigh), color, lineweight)
        img2 = cv.line(img2, (xlow, yhigh), (xhigh, yhigh), color, lineweight)
        if writerank:
            font = cv.FONT_HERSHEY_SIMPLEX
            img2 = cv.putText(img2, str(count), (xind+2, yind+16), font, 4, 0, 4, cv.LINE_AA)
        count += 1
        img2 = imutils.rotate(img2, -scoretable[ind][2])
        img2[img2 == 0] = img[img2 == 0]
    return img2


def draw_samples_3(img, scoretable, shape, writerank=False, lineweight=5):
    """
    Created 2019-05-08
    Draws squares at likely mark locations
    :param img: An image from which to draw samples
    :param scoretable: A table of locations, angles, and scores for marks in the image to be extracted
    :param shape: shape of the sample to be extracted
    :param writerank: if True, write the number indicating how high the critscore was compared to other boxes in img
    :param lineweight: weight of lines to draw
    :return samples: a list of samples from the image
    :return img: an image with squares drawn at the saved feature locations
    :return scoretable: A list of the positions, scores, and angles of each square drawn
    """
    img2 = copy.copy(img)
    count = 0
    color = (255, 255, 255)
    diag = np.sqrt(shape[0]**2 + shape[1]**2)
    for ind in range(len(scoretable)):
        xind = int(scoretable[ind][0])
        yind = int(scoretable[ind][1])
        x1 = int(xind - shape[1] / 2 * np.cos(scoretable[ind][2]) + shape[0] / 2 * np.sin(scoretable[ind][2]))
        x2 = int(xind + shape[1] / 2 * np.cos(scoretable[ind][2]) + shape[0] / 2 * np.sin(scoretable[ind][2]))
        x3 = int(xind + shape[1] / 2 * np.cos(scoretable[ind][2]) - shape[0] / 2 * np.sin(scoretable[ind][2]))
        x4 = int(xind - shape[1] / 2 * np.cos(scoretable[ind][2]) - shape[0] / 2 * np.sin(scoretable[ind][2]))
        y1 = int(yind - shape[0] / 2 * np.cos(scoretable[ind][2]) - shape[1] / 2 * np.sin(scoretable[ind][2]))
        y2 = int(yind - shape[0] / 2 * np.cos(scoretable[ind][2]) + shape[1] / 2 * np.sin(scoretable[ind][2]))
        y3 = int(yind + shape[0] / 2 * np.cos(scoretable[ind][2]) + shape[1] / 2 * np.sin(scoretable[ind][2]))
        y4 = int(yind + shape[0] / 2 * np.cos(scoretable[ind][2]) - shape[1] / 2 * np.sin(scoretable[ind][2]))
        img2 = cv.line(img2, (x1, y1), (x2, y2), color, lineweight)
        img2 = cv.line(img2, (x2, y2), (x3, y3), color, lineweight)
        img2 = cv.line(img2, (x3, y3), (x4, y4), color, lineweight)
        img2 = cv.line(img2, (x4, y4), (x1, y1), color, lineweight)
        if writerank:
            font = cv.FONT_HERSHEY_SIMPLEX
            img2 = cv.putText(img2, str(count), (xind+2, yind+16), font, 4, 0, 4, cv.LINE_AA)
        count += 1
    return img2


def extract_samples_2(img, scoretable, shape):
    """
    Created 2019-01-28
    Extracts samples of likely mark locations from an image
    Assumes the indexes apply after the rotation
    :param img: An image from which to draw samples
    :param scoretable: A table of locations, angles, and scores for marks in the image to be extracted
    :param shape: shape of the sample to be extracted
    :return samples: a list of samples from the image
    """
    samples = []
    for ind in range(len(scoretable)):
        xind = scoretable[ind][0]
        yind = scoretable[ind][1]
        xlow, xhigh, ylow, yhigh = high_low(xind, yind, int(shape[1]/2), int(shape[0]/2), img.shape)
        img2 = imutils.rotate(img, scoretable[ind][2])
        s = img2[ylow:yhigh, xlow:xhigh]
        samples.append(s)
    return samples


def extract_samples_3(img, scoretable, shape, add_one_y=0):
    """
    Created 2019-03-27
    Extracts samples of likely mark locations from an image
    Assumes the indexes apply before the rotation
    :param img: An image from which to draw samples
    :param scoretable: A table of locations, angles, and scores for marks in the image to be extracted
    :param shape: shape of the sample to be extracted
    :param add_one_y: this can be a useful feature if the height of the sample to be extracted needs to be adjusted
    by 1. Since the sample size is changed by rounding down after dividing by two, increasing the sample size by one
    can not be done directly by changing shape.
    :return samples: a list of samples from the image
    """
    samples = []
    diag = int(np.sqrt(shape[0]**2 + shape[1]**2)/2) + 1
    for ind in range(len(scoretable)):
        xind = scoretable[ind][0]
        yind = scoretable[ind][1]
        xlow, xhigh, ylow, yhigh = high_low(xind, yind, diag, diag, img.shape)
        img2 = imutils.rotate(img[ylow:yhigh, xlow:xhigh], scoretable[ind][2])
        s = img2[int(diag-shape[0]/2):int(diag+shape[0]/2)+add_one_y,
                 int(diag-shape[1]/2):int(diag+shape[1]/2)]
        samples.append(s)
    return samples


def high_low(x, y, w, h, shape):
    """
    Computes upper and lower index bounds to define a sub-array to extract from
    a larger array
    :param x: center column index of region to be considered
    :param y: center row index of region to be considered
    :param w: half-width of region to be considered
    :param h: half-height of region to be considered
    :param shape: [n_rows of larger array, n_columns of larger array]
    :return:
    """
    ylow = int(max(0, y - h))
    yhigh = int(min(y + h, shape[0]))
    xlow = int(max(0, x - w))
    xhigh = int(min(x + w, shape[1]))
    return xlow, xhigh, ylow, yhigh


def fcn_mark_im_ind(im_ind_raw, n_img):
    """
    Calculates the mark an image index based on the total image index
    For example, im_ind_raw is 52 and n_img is 30, then mark_ind is 1
    and im_ind is 22
    """
    mark_ind = int(im_ind_raw/n_img)
    im_ind = im_ind_raw - mark_ind * n_img
    return mark_ind, im_ind
