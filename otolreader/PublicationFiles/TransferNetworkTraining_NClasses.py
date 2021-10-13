#!/usr/bin/env python3.6

# This script is designed to train set of classification networks that are also used to select samples,
# as well as a second set of networks to distinguish between marked and unmarked samples
# The two sets of networks can then be used for cross validation.
from __future__ import print_function

import os
import json
import pickle
import sys
import traceback
import numpy as np
from tensorflow import keras
import tensorflow as tf
import argparse

np.random.seed(1)
n_cvfolds = 1
# -----------------------------------
# Instantiating argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    # parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()

# -----------------------------------
# These are the paths to where SageMaker mounts interesting things in your container.

prefix = '../../../OtolReader-Publication'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'models-temp')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# file mode, the input files are copied to the directory specified here.
channel_name = 'training'
training_path = os.path.join('../../..', 'TrainingSampleDatabase')


def classification_model_init(tr):
    # ------------------ Initialize the model ----------------------------------
    model = keras.Sequential([
        keras.layers.Conv1D(256, 3, activation=tf.nn.relu, input_shape=tr.shape[1:]),
        keras.layers.Conv1D(128, 9, activation=tf.nn.relu),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def binary_model_init(tr):
    model = keras.Sequential([
        keras.layers.Conv1D(256, 3, activation=tf.nn.relu, input_shape=tr.shape[1:]),
        keras.layers.Conv1D(128, 9, activation=tf.nn.relu),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def make_fold_train_array(train_X, train_y, foldinds, cv_order=None):
    """
    Develops training and validation arrays and labels.
    cv_order is an optional argument. If passed, the code only extracts
    items from train_X with indexes that are in cv_order
    """
    tr = []
    val = []
    tr_labs = []
    val_labs = []
    for im_ind in range(len(train_X)):
        if im_ind in foldinds:
            val.extend(train_X[im_ind])
            val_labs.extend([train_y[im_ind], ] * len(train_X[im_ind]))
        else:
            tr.extend(train_X[im_ind])
            tr_labs.extend([train_y[im_ind], ] * len(train_X[im_ind]))
    if len(val) > 0:
        val = np.array(val)
        val_labs = np.array(val_labs, dtype=int)
        val = np.expand_dims(val, axis=2)
    tr = np.array(tr)
    tr_labs = np.array(tr_labs, dtype=int)
    tr = np.expand_dims(tr, axis=2)
    return tr, tr_labs, val, val_labs


def calc_fold_inds(n_cvfolds, cv_order, cvind):
    if n_cvfolds == 1:
        nims_fold = 0
    else:
        nims_fold = int(len(cv_order) / n_cvfolds)
    foldinds = cv_order[cvind * nims_fold:(cvind + 1) * nims_fold]
    return foldinds


def classification_model_trainer(training_array_path, n_cvfolds, n_img, cv_order):
    with open(training_array_path, 'rb') as f:
        print('---------------------Training file name: ', training_array_path)
        training_temp = pickle.load(f)
    n_class = int(len(cv_order) / n_img)
    train_y = []
    train_X = []
    mark_count = 0
    for mark, samps in training_temp.items():
        for im_samps in samps:
            train_X.append(im_samps)
            train_y.append(mark_count)
        mark_count += 1
    train_X = np.array(train_X)
    train_y = np.array(train_y, dtype=int)
    for cvind in range(n_cvfolds):
        foldinds = calc_fold_inds(n_cvfolds, cv_order, cvind)
        tr, tr_labs, val, val_labs = make_fold_train_array(train_X, train_y, foldinds, cv_order)
        print('tr shape:', tr.shape)

        # ------------------ Initialize the model ----------------------------------
        model = classification_model_init(tr)
        if len(val) > 0:
            history = model.fit(tr, tr_labs, epochs=10, verbose=0, validation_data=(val, val_labs))
        else:
            history = model.fit(tr, tr_labs, epochs=10, verbose=0)
        # save the model
        # tf.contrib.saved_model.save_keras_model(model, args.model_dir)
        n_class_path = os.path.join(model_path, str(n_class) + ' classes')
        if str(n_class) + ' classes' not in os.listdir(model_path):
            os.mkdir(n_class_path)
        model.save(os.path.join(n_class_path, 'classnet_' + str(cvind) + '.h5'))
        with open(os.path.join(n_class_path, 'classhist_' + str(cvind) + '.h5'), 'wb') as f:
            pickle.dump(history.history, f)
    print('Training complete.')


def binary_model_trainer(training_array_path, n_cvfolds, n_img, cv_order):
    with open(training_array_path, 'rb') as f:
        print('---------------------Training file name: ', training_array_path)
        training_temp = pickle.load(f)
    n_class = int(len(cv_order) / n_img)
    train_y = []
    train_X = []
    mark_count = 0
    for mark, samps in training_temp.items():
        for im_samps in samps:
            train_X.append(im_samps)
            if mark == 'None':
                train_y.append(0)
            else:
                train_y.append(1)
        mark_count += 1
    train_X = np.array(train_X)
    train_y = np.array(train_y, dtype=int)
    for cvind in range(n_cvfolds):
        foldinds = calc_fold_inds(n_cvfolds, cv_order, cvind)
        tr, tr_labs, val, val_labs = make_fold_train_array(train_X, train_y, foldinds, cv_order)
        print('tr shape:', tr.shape)
        model = binary_model_init(tr)
        if len(val) > 0:
            history = model.fit(tr, tr_labs, epochs=10, verbose=0, validation_data=(val, val_labs))
        else:
            history = model.fit(tr, tr_labs, epochs=10, verbose=0)
        # save the model
        # tf.contrib.saved_model.save_keras_model(model, args.model_dir)
        model.save(os.path.join(model_path, str(n_class) + ' classes', 'binarynet_' + str(cvind) + '.h5'))
        with open(os.path.join(model_path, str(n_class) + ' classes', 'binaryhist_' + str(cvind) + '.h5'), 'wb') as f:
            pickle.dump(history.history, f)
    print('Training complete.')


# The function to execute the training.
def train():
    print('Starting the training.')
    # -----------------------------------
    # These are constants for now
    n_img = 30  # Number of images per mark for training
    stepsize = 10  # step size for second derivative convolution
    avg_window = 20  # window size for moving average convolution
    pdir = 'TrainingSamples'  # Directory of training samples
    xz = np.linspace(0, 799, 800)  # linear array with length of samples
    inds = np.linspace(0, 150, 151, dtype=int) * 5  # Indexes of values to save from samples
    # Set the cross val order:
    for n_class in range(2, 6):  # Number of classes
        # these if statements ensure that the "None" class is only included if n_class is 5
        if n_class == 2 or n_class == 5:
            cv_order = np.linspace(0, n_img * n_class - 1, n_img * n_class, dtype=int)
        else:
            cv_order = np.linspace(0, n_img * 2 - 1, n_img * 2, dtype=int)
            cv_order2 = np.linspace(n_img * 3, n_img * (n_class + 1) - 1, n_img * n_class, dtype=int)
            cv_order = np.concatenate([cv_order, cv_order2])

        np.random.shuffle(cv_order)
        if str(n_class) + ' classes' not in os.listdir(model_path):
            os.mkdir(os.path.join(model_path, str(n_class) + ' classes'))
        with open(os.path.join(model_path, str(n_class) + ' classes', 'cvorder.p'), 'wb') as f:
            pickle.dump(cv_order, f)
        try:
            # Read in any hyperparameters that the user passed with the training job
            # with open(param_path, 'r') as tc:
            #     trainingParams = json.load(tc)
            input_files = os.listdir(training_path)
            if len(input_files) == 0:
                raise ValueError(('There are no files in {}.\n' +
                                  'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                                  'the data specification in S3 was incorrectly specified or the role specified\n' +
                                  'does not have permission to access the data.').format(training_path, channel_name))
            cl_tr_path = os.path.join(training_path, 'publication_classification_training_array.p')
            classification_model_trainer(cl_tr_path, n_cvfolds, n_img, cv_order)
            bi_tr_path = os.path.join(training_path, 'publication_binary_training_array.p')
            binary_model_trainer(bi_tr_path, n_cvfolds, n_img, cv_order)
            print('Training complete.')
        except Exception as e:
            # Write out an error file. This will be returned as the failureReason in the
            # DescribeTrainingJob result.
            trc = traceback.format_exc()
            with open(os.path.join(output_path, 'failure'), 'w') as s:
                s.write('Exception during training: ' + str(e) + '\n' + trc)
            # Printing this causes the exception to be in the training job logs, as well.
            print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
            # A non-zero exit code causes the training job to be marked as Failed.
            sys.exit(255)


if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)

