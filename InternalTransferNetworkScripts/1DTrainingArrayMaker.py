from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import os
import pickle
import sys
import cv2 as cv
sys.path.append('..')
import OtolithAnalysis as oa
import time
mark_list = ['3,5H10', '1,6H', 'None', '6,2H', '4n,2n,2H']


# -------------------Generate training array------------------------
n_img = 30  # Number of images per mark for training
stepsize = 10  # step size for second derivative convolution
avg_window = 20  # window size for moving average convolution
ngen = 20  # number of synthetic samples to generate per natural sample
pdir = 'TrainingSamples_2'  # Directory of training samples
counter, mark_count = 0, 0  # for counting samples and marks, respectively
xz = np.linspace(0, 799, 800)  # linear array with length of samples
inds = np.linspace(0, 150, 151, dtype=int) * 5  # Indexes of values to save from samples
#  max_s_per_im = 120  # maximum number of samples per image
training = [[] for i in range(n_img*len(mark_list))]
training_labels = np.zeros(n_img * len(mark_list), dtype=int)
training_array = None  # "SavedTrainingArrays/training_aug_rot10height02flip_40.p"

def im_proc(mark_type, im_ind):
    training = []
    # print("Currently evaluating image", im_ind, "mark", mark_type)
    for s_name in os.listdir(os.path.join(pdir, mark_type, str(im_ind))):
        fpath = os.path.join(pdir, mark_type, str(im_ind), s_name)
        with open(fpath, 'rb') as f:
            s = np.array(pickle.load(f), dtype=np.float32)
        out = generator.flow(np.expand_dims(np.expand_dims(s, axis=3), axis=0),
                             np.array([mark_count]))
        for _ in range(ngen):
            s = out.next()[0][0, :, :, 0]
            coeffs = np.polyfit(xz[:len(s)], np.mean(s, 1), deg=1)
            for ind in range(s.shape[1]):
                s[:, ind] = s[:, ind] - np.polyval(coeffs, xz[:s.shape[0]])
            s = s / np.std(s) * 20
            s = np.mean(s, 1)
            d2ydx2 = s[:-stepsize*2] + s[stepsize*2:] - 2 * s[stepsize:-stepsize]
            d2ydx2 = np.convolve(d2ydx2, np.ones(avg_window) / avg_window, mode='valid')
            if len(d2ydx2) != 761:
                print(im_ind, len(d2ydx2))
            training.append(d2ydx2[inds])
    return [im_ind, np.array(training)]


print("Generating training array")
# Generates augmented samples rotated randomly by -10 to +10 degrees
generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=10, height_shift_range=0.02, vertical_flip=True)
for mark_type in mark_list:
    temp = Parallel(n_jobs=30)(delayed(im_proc)(mark_type, im_ind) for im_ind in range(n_img))
    for im_ind in range(n_img):
        for ind in range(len(temp)):
            if im_ind == temp[ind][0]:
                training[mark_count].append(temp[ind][1])
    print("Completed training set for mark " + mark_list[mark_count])
    mark_count += 1
for mark_ind in range(len(mark_list)):
    print(mark_list[mark_ind] + ' samples: ', np.sum(training_labels == mark_ind))
training_labels = np.array(training_labels, dtype=int)
training = np.array(training)
with open("SavedSampleSelectorTrainingArrays/2019_03_27_training_aug_rot10height02flip_20.p", 'wb') as f:
    pickle.dump(training, f)
