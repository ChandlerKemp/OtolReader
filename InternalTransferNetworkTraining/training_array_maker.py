from __future__ import absolute_import, division, print_function
from tensorflow import keras
import numpy as np
from joblib import Parallel, delayed
import os
import pickle
import OtolReader.OtolithAnalysis as oa


def im_proc(mark_type, im_ind, generator, pdir, stepsize, avg_window, ngen, inds, xz, max_s_per_im):
    """
    Loads a sample file, generates a set of random variations of the file, computes a reduced 1D version of all
    variations in the set, and returns all of the reduced vectors in an array along with the image index.
    :param mark_type: name of mark being considered
    :param im_ind: index of the image from which the sample will be pulled
    :param generator: a tensorflow generator object to create synthetic samples
    :param pdir: directory containing samples to be loaded
    :param stepsize: stepsize to use for second derivative calculations
    :param avg_window: kernel size to use for smoothing
    :param ngen: number of synthetic samples to generate per natural sample
    :param inds: array of indexes to store
    :param xz: aray of indexes to use in a linear fit to remove trends
    :param max_s_per_im: maximum number of samples to pull from one image
    :return: index of the image from which the sample was pulled, array of training samples
    """
    training = []
    # print("Currently evaluating image", im_ind, "mark", mark_type)
    for s_name in os.listdir(os.path.join(pdir, mark_type, str(im_ind)))[:max_s_per_im]:
        fpath = os.path.join(pdir, mark_type, str(im_ind), s_name)
        with open(fpath, 'rb') as f:
            s = np.array(pickle.load(f), dtype=np.float32)
        out = generator.flow(np.expand_dims(np.expand_dims(s, axis=s.ndim), axis=0),
                             np.array([mark_type]))
        for _ in range(ngen):
            s = out.next()[0][0, :, :, 0]
            s = oa.finder_network.fcn_smoothed_d2ydx2(s, stepsize=stepsize, avg_window=avg_window, xz=xz, inds=inds)
            training.append(s)
    return np.array(training)


def training_array_generator(file_out, n_img=30, pdir=None, stepsize=10, avg_window=20, ngen=20, inds=None,
                             max_s_per_im=30, mark_list=['None']):
    """
    :param file_out: path at which to store the output file
    :param n_img: the number of images in each class to use in training
    :param pdir: directory of samples to load
    :param stepsize: the stepsize to use when computing the numberical second derivative
    :param avg_window: The kernel size to use in smoothing
    :param ngen: the number of sythetic samples to generate per natural sample
    :param inds: the set of indexes to store in the output array
    :param max_s_per_im: maximum number of samples to pull from one image
    :param mark_list: list of mark labels (must be the same as sample directory labels)
    """
    if inds is None:
        inds = np.linspace(0, 150, 151, dtype=int) * 5  # Indexes of values to save from samples
    xz = np.linspace(0, 799, 800)  # linear array with length of samples
    #  max_s_per_im = 120  # maximum number of samples per image
    training = {}
    training_array = None  # "SavedTrainingArrays/training_aug_rot10height02flip_40.p"
    print("Generating training array")
    # Generates augmented samples rotated randomly by -10 to +10 degrees
    generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                             height_shift_range=0.02,
                                                             vertical_flip=True)
    args = [generator, pdir, stepsize, avg_window, ngen, inds, xz, max_s_per_im]
    samp_counts = {}
    for mark in mark_list:
        training[mark] = Parallel(n_jobs=n_img)(delayed(im_proc)(mark, im_ind, *args) for im_ind in range(n_img))
        samp_counts[mark] = 0
        for im_ind in range(n_img):
            samp_counts[mark] += training[mark][im_ind].shape[0]
        print("Completed training set for mark " + mark)
    for mark in mark_list:
        print(mark + ' samples: ', samp_counts[mark])
    with open(file_out, 'wb') as f:
        pickle.dump(training, f)
