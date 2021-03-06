import numpy as np
import imutils
from .feature_functions import high_low


def finder(img_in, model, angles=[0], shape=None, percentile=99, x_step=120, y_step=120, slength=151,
           print_status=False):
    """
    Locates samples in an image that are most likely to be a mark and returns their location (x, y, angle) and scores
    in a table.
    :param img_in: a grayscale image in a numpy array
    :param model: A trained tensor flow keras model object
    :param angles: the set of angles at which to search the image
    :param shape: dimensions of the samples to consider
    :param percentile: the percentile of samples to keep
    :param x_step: number of columns between samples (center to center)
    :param y_step: number of rows between samples (center to center)
    :return maxscores: an array of of high scores and their locations (each row stores: x, y, angle, f_fft,
    where x and y are the locations of the center of the sample, angle is the angle that the image was rotated before
    sampling and f_fft is the fft score of the sample). The number of rows depends on the size of the image, x_step,
    y_step, the length of angles and percentile.
    """
    if shape is None:
        shape = np.array([800, 240])
    score_table = []
    xz = np.linspace(0, shape[0] - 1, shape[0], dtype=int)
    diag = int(np.sqrt(shape[0]**2 + shape[1]**2)/2) + 1
    nsteps = int((img_in.shape[1] - 2 * diag) / x_step) + 1
    counter = 1
    for x in range(diag, img_in.shape[1]-diag, x_step):
        if print_status:
            print("Evaluating step " + str(counter) + " of " + str(nsteps))
        counter += 1
        for y in range(diag, img_in.shape[0]-diag, y_step):
            xlow, xhigh, ylow, yhigh = high_low(x, y, diag, diag, img_in.shape)
            samples = np.zeros([len(angles), slength])
            for ang_ind in range(len(angles)):
                img2 = imutils.rotate(img_in[ylow:yhigh, xlow:xhigh], angles[ang_ind])
                s = img2[int(diag - shape[0] / 2):int(diag + shape[0] / 2),
                    int(diag - shape[1] / 2):int(diag + shape[1] / 2)]
                samples[ang_ind, :] = fcn_smoothed_d2ydx2(s)
            scores = model.predict(np.expand_dims(samples, axis=2))
            scores = scores[:, 1]
            score_table.append([x, y, angles[np.argmax(scores)], np.max(scores)])
    score_table = np.array(score_table)
    maxscores = score_table[score_table[:, 3] >= np.percentile(score_table[:, 3], percentile), :]
    return maxscores


def fcn_smoothed_d2ydx2(s, stepsize=10, avg_window=20, saveind_step=5, xz=None, inds=None):
    """
    computes the smoothed second derivative of s after collapsing to one dimension 0
    :param s: input array
    :param stepsize: step size to use in computing the second derivative
    :param avg_window: window size to use in applying moving average
    :param saveind_step: stepsize between saved indexes. For example, saveind_step=5 implies saving indexes 0, 5, 10,...
    :param xz: optional input array of indexes
    :param inds: if specified, saveind_step is overridden and indexes in this array are saved.
    :return d2ydx2: the smoothed second derivative
    """
    # Note! the -1 here is a relic from early iterations of this function
    if inds is None:
        n_steps = int((s.shape[0] - 2 * stepsize - avg_window + 1)/saveind_step) - 1
        inds = np.linspace(0, n_steps-1, n_steps, dtype=int) * saveind_step
    if xz is None:
        xz = np.linspace(0, len(s)-1, len(s), dtype=int)
    coeffs = np.polyfit(xz[:len(s)], np.mean(s, 1), deg=1)
    sout = np.zeros(s.shape)
    for ind in range(s.shape[1]):
        sout[:, ind] = s[:, ind] - coeffs[1] - coeffs[0] * xz[:s.shape[0]]
    sout = sout / np.std(sout) * 20
    sout = np.mean(sout, 1)
    sout = sout[:-stepsize * 2] + sout[stepsize * 2:] - 2 * sout[stepsize:-stepsize]
    sout = np.convolve(sout, np.ones(avg_window) / avg_window, mode='valid')
    return sout[inds]
