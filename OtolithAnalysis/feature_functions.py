import numpy as np
import imutils
import cv2 as cv
import copy
import os
from fpdf import FPDF


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


def feature_scores(img, feature, print_status=True, goodinds=None, angles_in=np.linspace(0, 170, 18)):
    """
    Returns an array with shape img.shape with scores for the feature
    Note: to go from the fitvalue coordinates to image coordinates,
    add diag//2
    :param img: a gray scale image
    :param feature: a feature
    :param printstatus: if True, print the status of the function during execution
    :param angles_in: iterable set of angles to consider
    :return fitvalue: array of feature fit scores
    :return saveposition: angle of best score at each position
    :return maxscore: best score for the feature
    :return angles: angle of fit to achieve fitvalue at each position
    """
    maxscore = 0
    fheight, fwidth = feature.shape
    wdiag = int(np.max(fwidth * np.abs(np.cos(angles_in * np.pi / 180)) +
                       fheight * np.abs(np.sin(angles_in * np.pi / 180))))+1
    hdiag = int(np.max(fheight * np.abs(np.cos(angles_in * np.pi / 180)) +
                       fwidth * np.abs(np.sin(angles_in * np.pi / 180))))+1
    fitvalue = np.zeros(np.array(img.shape) - np.array([hdiag, wdiag]) + 1)
    angles = np.zeros(np.array(img.shape) - np.array([hdiag, wdiag]) + 1)
    for angle in angles_in:
        ftemp = imutils.rotate_bound(feature, angle)
        ftemp -= np.mean(ftemp)
        ht, wid = ftemp.shape
        for xind in range(wdiag//2, img.shape[1]-(wdiag-1)//2):
            for yind in range(hdiag//2, img.shape[0] - (hdiag-1)//2):
                if goodinds is None or goodinds[yind - hdiag//2, xind - wdiag//2]:
                    temp = img[yind - ht//2:yind - ht//2 + ht,
                               xind - wid//2:xind - wid//2 + wid]
                    score = np.sum(ftemp * temp)
                    if score > maxscore:
                        saveposition = [angle, xind, yind]
                        maxscore = score
                    if score > fitvalue[yind - hdiag//2, xind - wdiag//2]:
                        fitvalue[yind - hdiag//2, xind - wdiag//2] = score
                        angles[yind - hdiag//2, xind - wdiag//2] = angle
                else:
                    fitvalue[yind - hdiag//2, xind - wdiag//2] = np.nan
        if print_status and np.mod(angle, 30) == 0:
            print("Completed evaluation of angle {:0.0f}".format(angle))
    return fitvalue, saveposition, maxscore, angles


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


def make_pdf(samples, fname_out):
    """
    Created 2019-01-28
    Generates a pdf of the images in "samples", with one image per page
    :param samples: a list of image samples to be placed in a pdf document
    :param fname_out: file name at which to store the resulting pdf
    :return: none
    """
    ind = 0
    pdf = FPDF(unit="pt", format=[samples[0].shape[1], samples[0].shape[0]])
    os.mkdir('temp_images')
    for im in samples:
        ind += 1
        fname = 'temp_images/' + str(ind) + '.jpg'
        cv.imwrite(fname, im)
        pdf.add_page()
        pdf.image(fname, 0, 0)
    for f in os.listdir('temp_images'):
        os.remove('temp_images/' + f)
    os.rmdir('temp_images')
    _ = pdf.output(fname_out, "F")


def extract_samples(img, scoretable, shape):
    """
    Extracts samples of likely mark locations from an image
    :param img: An image from which to draw samples
    :param scoretable: A table of locations, scores and angles for marks in the image to be extracted
    :param shape: shape of the sample to be extracted
    :return samples: a list of samples from the image
    """
    samples = []
    diag = int(np.sqrt(np.sum(shape**2)))+1
    w = int(diag/2) + 1
    for ind in range(len(scoretable)):
        xind = scoretable[ind][0]
        yind = scoretable[ind][1]
        xlow, xhigh, ylow, yhigh = high_low(xind, yind, w, w, img.shape)
        s = img[ylow:yhigh, xlow:xhigh]
        s = imutils.rotate_bound(s, -scoretable[ind][-1][0])
        center = np.array(np.array(s.shape)/2, dtype=int)
        yind = center[0]
        xind = center[1]
        w2 = int(shape[1] / 2)
        h2 = int(shape[0] / 2)
        xlow, xhigh, ylow, yhigh = high_low(xind, yind, w2, h2, s.shape)
        s = s[ylow:yhigh, xlow:xhigh]
        samples.append(s)
    return samples


def draw_samples(img, scores, angles, criteria, shape, fshape=[16, 10],
                 angles_in=np.linspace(0, 170, 18), writerank=False):
    """
    Draws non-overlapping squares at likely mark locations
    :param img: A raw image
    :param scores: An of array of scores from a list of features, dimensions of [Nfeatures, img.shape-feature.shape]
    :param angles: An array of angles from a list of features, dimensions of [Nfeatures, img.shape-feature.shape]
    :param criteria: the cutoff percentile at which to include points
    :param shape: size of square to be drawn
    :param fshape: shape of feature used to generate scores
    :param angles_in: set of angles used to generate scores
    :param writerank: if True, write the number indicating how high the critscore was compared to other boxes in img
    :return img: an image with squares drawn at the saved feature locations
    :return scoretable: A list of the positions, scores, and angles of each square drawn
    """
    fheight, fwidth = fshape
    wdiag = int(np.max(fwidth * np.abs(np.cos(angles_in * np.pi / 180)) +
                       fheight * np.abs(np.sin(angles_in * np.pi / 180))))+1
    hdiag = int(np.max(fheight * np.abs(np.cos(angles_in * np.pi / 180)) +
                       fwidth * np.abs(np.sin(angles_in * np.pi / 180))))+1
    scoretable = []
    font = cv.FONT_HERSHEY_SIMPLEX
    cond = np.ones(scores[0].shape, dtype=bool)
    size = max(shape)
    for ind in range(len(scores)):
        goodinds = np.isfinite(scores[ind])
        cond[np.isnan(scores[ind])] = False
        cond[goodinds] = cond[goodinds] & (scores[ind][goodinds] >=
                                           np.percentile(scores[ind][goodinds],
                                                         criteria[ind]))
    critscore = cond * np.nansum(scores, axis=0)
    count = 1
    while np.max(critscore) > 0:
        maxpos = np.argmax(critscore)
        xind = np.mod(maxpos, critscore.shape[1])
        yind = int((maxpos - xind) / critscore.shape[1])
        xmin = max(0, xind - size - size//2)
        ymin = max(0, yind - size - size//2)
        xmax = min(critscore.shape[1], xind + size + (size-1)//2)
        ymax = min(critscore.shape[0], yind + size + (size-1)//2)
        critscore[ymin:ymax, xmin:xmax] = 0
        # Note: to go from critscore indexes to image indexes,
        # add wdiag//2 or hdiag//2 as the case may be. To find the top left corner of the
        # rectangle to be drawn in the image, subtract the same values.
        # Therefore, the top left corner of the feature in the image
        # has the same index as the center of the feature in critscore
        img = cv.line(img, (xind, yind), (xind + size, yind), (0, 0, 0), 5)
        img = cv.line(img, (xind, yind), (xind, yind + size), (0, 0, 0), 5)
        img = cv.line(img, (xind + size, yind),
                      (xind + size, yind + size), (0, 0, 0), 5)
        img = cv.line(img, (xind, yind + size),
                      (xind + size, yind + size), (0, 0, 0), 5)
        scoretable.append([xind + wdiag//2, yind + hdiag//2, scores[:, yind, xind], angles[:, yind, xind]])
        if writerank:
            img = cv.putText(img, str(count), (xind+2, yind+16), font, 1, 255, 2, cv.LINE_AA)
        count += 1
    return img, scoretable


def fcn_mark_im_ind(im_ind_raw, n_img):
    """
    Calculates the mark an image index based on the total image index
    For example, im_ind_raw is 52 and n_img is 30, then mark_ind is 1
    and im_ind is 22
    """
    mark_ind = int(im_ind_raw/n_img)
    im_ind = im_ind_raw - mark_ind * n_img
    return mark_ind, im_ind
