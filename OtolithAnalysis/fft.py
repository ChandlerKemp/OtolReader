import numpy as np
import imutils
from .feature_functions import high_low


def fft_score(sample, xz=None, s_e=np.array([400, 240]),
              stepsize=10, wave_range=None, std_var=True):
    """
    Created 2019-01-18 (approx)
    Modified 2019-01-28
    Computes the fft feature score for a sample
    The score is calculated in a few steps:
    1.) sum the sample along axis 1 to produce a 1d array
    2.) subtract the linear trend and mean from sample
    3.) if std_var is true, normalize the variance of the array sample and recompute the 1d array
    4.) compute the numerical second derivative of the 1d array
    5.) compute the fft of the 1d array
    6.) sum over wave_range in the fft and return the result
    :param sample: an image (often a small subsection of an image)
    :param xz: the length of the 1d sample to which the fft will be applied
    :param s_e: dimension of the sample
    :param stepsize: step size to use in computing the second derivative
    :param wave_range: the range of indexes to sum in the fft for the final score
    :param std_var: if true, normalize the variance of the sample
    :return f_fft: the fft feature score for the sample
    """
    if xz is None:
        xz = np.linspace(0, s_e[0] - 1, s_e[0], dtype=int)
    if wave_range is None:
        wave_range = range(9, 12)
    s = np.array(sample, dtype=np.float32)
    if std_var:
        s = (s - np.mean(s)) * 50 / np.std(s)
    s = np.sum(s, 1)
    coeffs = np.polyfit(xz[:len(s)], s, 1)
    s = s - np.polyval(coeffs, xz[:len(s)])
    d2ydx2 = s[:-stepsize*2] + s[stepsize*2:] - 2 * s[stepsize:-stepsize]
    temp = np.abs(np.fft.fft(d2ydx2))
    if type(wave_range) is str:
        if wave_range.lower() == 'spectrum':
            f_fft = temp
    else:
        f_fft = np.sum(temp[wave_range])
    return f_fft


def fft_finder(img_in, angles=[0], s_e=None, wave_range=None, percentile=99, x_step=30, y_step=200):
    """
    Locates samples in an image that have a high fft score and returns their location (x, y, angle) and scores in a
    table.
    :param img_in: a grayscale image in a numpy array
    :param angles: the set of angles at which to search the image
    :param s_e: dimensions of the samples to consider
    :param wave_range: list of indexes to include in the fft score
    :param percentile: the percentile of samples to keep
    :param x_step: number of columns between samples (center to center)
    :param y_step: number of rows between samples (center to center)
    :return maxscores: an array of of high scores and their locations (each row stores: x, y, angle, f_fft,
    where x and y are the locations of the center of the sample, angle is the angle that the image was rotated before
    sampling and f_fft is the fft score of the sample). The number of rows depends on the size of the image, x_step,
    y_step, the length of angles and percentile.
    """
    if s_e is None:
        s_e = np.array([400, 240])
    if wave_range is None:
        wave_range = range(10, 12)
    score_table = []
    xz = np.linspace(0, s_e[0] - 1, s_e[0], dtype=int)
    for angle in angles:
        img = imutils.rotate(img_in, angle)
        for x in range(0, img.shape[1], x_step):
            for y in range(0, img.shape[0], y_step):
                xlow, xhigh, ylow, yhigh = high_low(x, y, int(s_e[1]/2), int(s_e[0]/2), img.shape)
                if img[ylow, xlow] == 0 or img[ylow, xhigh-1] == 0 \
                    or img[yhigh-1, xlow] == 0 or img[yhigh-1, xhigh-1] == 0:
                    continue
                f_fft = fft_score(img[ylow:yhigh, xlow:xhigh], xz, s_e, wave_range=wave_range, std_var=False)
                score_table.append([x, y, angle, f_fft])
    score_table = np.array(score_table)
    maxscores = score_table[score_table[:, 3] > np.percentile(score_table[:, 3], percentile), :]
    return maxscores
