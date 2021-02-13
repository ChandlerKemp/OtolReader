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
n_cvfolds = 10
# -----------------------------------
# Instantiating argument parser
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
#    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()

# -----------------------------------
# These are the paths to where SageMaker mounts interesting things in your container.

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# file mode, the input files are copied to the directory specified here.
channel_name = 'training'
training_path = os.path.join(input_path, channel_name)


def classification_model_trainer(input_files, mark_list, n_img):
    train_y = np.zeros(n_img * len(mark_list), dtype=int)
    for fpath in input_files:
        if 'aug_rot10' in fpath:
            break
    with open(fpath, 'rb') as f:
        print('---------------------', fpath)
        training_temp = pickle.load(f)
    train_X = []
    for mark_count in range(len(mark_list)):
        for im_ind in range(0, n_img):
            train_X.append(training_temp[mark_count * int(len(training_temp) / len(mark_list)) + im_ind])
            train_y[mark_count * n_img + im_ind] = mark_count
    train_X = np.array(train_X)
    train_y = np.array(train_y, dtype=int)
    tr = []
    tr_labs = []
    for im_ind in range(len(train_X)):
        tr.extend(train_X[im_ind])
        tr_labs.extend([train_y[im_ind], ] * len(train_X[im_ind]))
    tr = np.array(tr)
    tr_labs = np.array(tr_labs, dtype=int)
    tr = np.expand_dims(tr, axis=2)
    print('tr shape:', tr.shape)     
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
    # Here we only support a single hyperparameter. Note that hyperparameters are always passed in as
    # strings, so we need to do any necessary conversions.
    # max_leaf_nodes = trainingParams.get('max_leaf_nodes', None)
    # if max_leaf_nodes is not None:
    #     max_leaf_nodes = int(max_leaf_nodes)

    model.fit(tr, tr_labs, epochs=10, verbose=0)

    # save the model
    # tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    model.save(os.path.join(model_path, 'classnet_fulltraining.h5'))
    print('Training complete.')
    
    
def binary_model_trainer(input_files, mark_list, n_img):
    train_y = np.zeros(n_img * len(mark_list) + n_img, dtype=int)
    train_X = []
    for fpath in input_files:
        if 'aug_rot10' in fpath:
            with open(fpath, 'rb') as f:
                training_temp = pickle.load(f)
            for mark_count in range(len(mark_list)):
                for im_ind in range(0, n_img):
                    train_X.append(training_temp[mark_count * int(len(training_temp) / len(mark_list)) + im_ind])
                    if mark_count != 2:
                        train_y[mark_count * n_img + im_ind] = 1
        elif 'extra_none_samples' in fpath:
            with open(fpath, 'rb') as f:
                training_temp = pickle.load(f)
            train_X.append(training_temp[:1600])
    train_X = np.array(train_X)
    train_y = np.array(train_y, dtype=int)
    tr = []
    tr_labs = []
    for im_ind in range(len(train_X)):
        tr.extend(train_X[im_ind])
        tr_labs.extend([train_y[im_ind], ] * len(train_X[im_ind]))
    tr = np.array(tr)
    tr_labs = np.array(tr_labs, dtype=int)
    tr = np.expand_dims(tr, axis=2)
    print('tr shape:', tr.shape)     
    # ------------------ Initialize the model ----------------------------------
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
    # Here we only support a single hyperparameter. Note that hyperparameters are always passed in as
    # strings, so we need to do any necessary conversions.
    # max_leaf_nodes = trainingParams.get('max_leaf_nodes', None)
    # if max_leaf_nodes is not None:
    #     max_leaf_nodes = int(max_leaf_nodes)

    model.fit(tr, tr_labs, epochs=10, verbose=0)

    # save the model
    # tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    model.save(os.path.join(model_path, 'binarynet_fulltraining.h5'))
    print('Training complete.')
    

# The function to execute the training.
def train():
    print('Starting the training.')
    # -----------------------------------
    # These are constants for now
    mark_list = ['3,5H10', '1,6H', 'None', '6,2H', '4n,2n,2H']
    n_img = 30  # Number of images per mark for training
    stepsize = 10  # step size for second derivative convolution
    avg_window = 20  # window size for moving average convolution
    pdir = 'TrainingSamples'  # Directory of training samples
    xz = np.linspace(0, 799, 800)  # linear array with length of samples
    inds = np.linspace(0, 150, 151, dtype=int) * 5  # Indexes of values to save from samples
    # Set the cross val order:
    cv_order = np.linspace(0, n_img * len(mark_list) -1, n_img * len(mark_list), dtype=int)
    np.random.shuffle(cv_order)
    print("cv order:", cv_order)
    try:
        # Read in any hyperparameters that the user passed with the training job
        with open(param_path, 'r') as tc:
            trainingParams = json.load(tc)

        # Take the set of files and read them all into a single pandas dataframe
        input_files = [ os.path.join(training_path, file) for file in os.listdir(training_path) ]
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))
        classification_model_trainer(input_files, mark_list, n_img)
        binary_model_trainer(input_files, mark_list, n_img)
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

