from tensorflow import keras
import tensorflow as tf
import pickle
import os
import numpy as np


def make_transfer_module(model):
    model2 = keras.Sequential()
    model2.add(keras.layers.Conv1D(256, 3, weights=model.layers[0].get_weights(), activation=tf.nn.relu,
                                   trainable=False))
    model2.add(keras.layers.Conv1D(128, 9, activation=tf.nn.relu, weights=model.layers[1].get_weights(),
               trainable=False))
    model2.add(keras.layers.Flatten())
    model2.add(keras.layers.Dense(256, activation=tf.nn.relu, weights=model.layers[3].get_weights(), trainable=False))
    return model2


def make_none_marked_model(model_base):
    model = keras.Sequential()
    model.add(model_base)
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def make_train_array(fold_ind, nim_fold, basedir, samps_prefix, cvorder):
    """
    Generates an array of samples for training from within the training set. Recall that in this case,
    the order of samples loaded in the with statement below follows cvorder.

    :param fold_ind: index of the fold to consider
    :param nim_fold: number of images available for verification per fold
    :param basedir: directory containing an array of samples for transfer training
    :param samps_prefix: the prefix used in sample file names
    """

    with open(os.path.join(basedir, samps_prefix + str(fold_ind) + '.p'), 'rb') as f:
        train_array = pickle.load(f)
    tr = []
    tr_labs = []
    for t_ind in range(len(train_array)):
        if t_ind not in range(fold_ind * nim_fold, (fold_ind + 1) * nim_fold):
            tr.extend(train_array[t_ind])
            mark_ind, im_ind = fcn_mark_im_ind(cvorder[t_ind], 30)
            if mark_ind == 2:
                lab = 0
            else:
                lab = 1
            tr_labs.extend([lab for _ in range(len(train_array[t_ind]))])
    return tr, tr_labs, train_array


def fcn_mark_im_ind(im_ind_raw, n_img):
    """
    Calculates the mark an image index based on the total image index
    For example, im_ind_raw is 52 and n_img is 30, then mark_ind is 1
    and im_ind is 22
    """
    mark_ind = int(im_ind_raw/n_img)
    im_ind = im_ind_raw - mark_ind * n_img
    return mark_ind, im_ind


def make_trained_transfer_nets(pdir, model_prefix, nim_fold, sampledir, samps_prefix, cvorder, dirout):
    """
    Iterates across each fold, generates an updated binary network and saves the new network
    :param pdir: Directory containing trained binary models
    :param model_prefix: Prefix used to name pretrained binary model files
    :param nim_fold: number of images available for verification per fold
    :param sampledir: directory containing an array of samples for transfer training
    :param samps_prefix: the prefix used in sample file names
    :param cvorder: order of images called for cross validation
    :param dirout: directory in which to save retrained models
    """
    fmod_name_base = os.path.join(pdir, model_prefix)
    tf.config.run_functions_eagerly(True)
    if nim_fold == 0:
        n_fold = 1
    else:
        n_fold = int(cvorder / nim_fold)
    for fold_ind in range(n_fold):
        print("Currently evaluating fold number {}".format(fold_ind))
        mod = keras.models.load_model(fmod_name_base + str(fold_ind) + '.h5', compile=False)
        temp_model = make_transfer_module(mod)
        tr, tr_labs, train_array = make_train_array(fold_ind, nim_fold, sampledir, samps_prefix, cvorder)
        # tr consists of samples selected by the classification network
        # we now pad those samples so that half of the total samples are unmarked
        while np.mean(tr_labs) > 0.5:
            none_im_ind = np.random.randint(0, 30)
            # This makes sure that none samples are not drawn from the validation set
            while none_im_ind + 60 in cvorder[fold_ind * nim_fold:(fold_ind + 1) * nim_fold]:
                none_im_ind = np.random.randint(0, 30)
            tr_ind = np.where(cvorder == none_im_ind + 60)[0][0]
            tr.extend(train_array[tr_ind])
            tr_labs.extend([0 for _ in range(len(train_array[tr_ind]))])
        tr = np.array(tr)
        tr_labs = np.array(tr_labs, dtype=int)
        nm_model = make_none_marked_model(temp_model)
        history = nm_model.fit(np.expand_dims(tr, axis=2), tr_labs, epochs=10)
        nm_model.save(os.path.join(dirout, 'binarynet_' + str(fold_ind) + '.h5'))
        with open(os.path.join(dirout, 'binaryhist_' + str(fold_ind) + '.h5'), 'wb') as f:
            pickle.dump(history.history, f)
