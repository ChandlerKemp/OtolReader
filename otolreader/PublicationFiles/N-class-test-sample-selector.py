import pickle
from tensorflow import keras
import numpy as np
import multiprocessing
import os
from otolreader.InternalTransferNetworkTraining import classification_network_mark_score_table as cnmst
from otolreader.OtolithAnalysis import im_classifier as ic
import otolreader

# ----------------------------------------------Constants--------------------------
mark_list_in = ['3,5H10', '1,6H', 'None', '6,2H', '4n,2n,2H']
n_train_img = 30  # number of training images per class
n_img = 20  # Number of test images per class
n_img_train = 30  # Number of training images per class
start_ind = 30
n_fold = 1
nim_fold = 0  # number of validation images
n_class = 5
binary_model_prefix = 'binarynet_'
if n_fold > 1:
    im_per_fold = n_class * n_img / n_fold  # Number of images in each fold
else:
    im_per_fold = 0
imdir = "../../../Images/OtolithImages/"
pdir = os.path.join('..', '..', '..', 'OtolReader-Publication', 'Models', 'N-class-test-set',
                    '{:0.0f} classes'.format(n_class))
im_dir = os.path.join('..', '..', '..', 'Images', 'OtolithImages')
retrained_dirout = os.path.join(pdir, 'retrained_binary_net')
model = keras.models.load_model(os.path.join(pdir, 'classnet_0.h5'), compile=False)

if 'temp' not in os.listdir():
    os.mkdir('temp')

if 'temp_train' not in os.listdir():
    os.mkdir('temp_train')


def sctable_entry(im_index):
    mark_ind, im_ind = otolreader.OtolithAnalysis.feature_functions.fcn_mark_im_ind(im_index, n_img)
    impath = os.path.join(imdir, mark_list_in[mark_ind], str(im_ind + start_ind) + '.jpg')
    score_table, samples, scores = ic.find_marks(impath, model)
    with open(os.path.join('temp', str(im_index) + '.p'), 'wb') as f:
        pickle.dump([score_table, samples, scores], f)


def sctable_entry_train(im_index):
    mark_ind, im_ind = otolreader.OtolithAnalysis.feature_functions.fcn_mark_im_ind(im_index, n_img_train)
    impath = os.path.join(imdir, mark_list_in[mark_ind], str(im_ind) + '.jpg')
    score_table, samples, scores = ic.find_marks(impath, model)
    with open(os.path.join('temp_train', str(im_index) + '.p'), 'wb') as f:
        pickle.dump([score_table, samples, scores], f)


def results_sorter(dir_in, dir_out):
    score_table, samples, scores = [], [], []
    for ind in range(len(os.listdir(dir_in))):
        with open(os.path.join(dir_in, str(ind) + '.p'), 'rb') as f:
            res = pickle.load(f)
        score_table.append(res[0])
        samples.append(res[1])
        scores.append(res[2])

    with open(os.path.join(dir_out, 'score_table_0.p'), 'wb') as f:
        pickle.dump(score_table, f)

    with open(os.path.join(dir_out, 'scores_0.p'), 'wb') as f:
        pickle.dump(scores, f)

    with open(os.path.join(dir_out, 'samples_0.p'), 'wb') as f:
        pickle.dump(samples, f)
# ----------------------------------------Analysis------------------------
if __name__ == '__main__':
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(sctable_entry, range(n_img * len(mark_list_in)))

    results_sorter('temp', pdir)

    with multiprocessing.Pool(processes=10) as pool:
        pool.map(sctable_entry_train, range(n_img_train * len(mark_list_in)))

    retrain_dir = os.path.join(pdir, 'retrain samples')
    if not os.path.isdir(retrain_dir):
        os.mkdir(retrain_dir)
    results_sorter('temp_train', retrain_dir)

    if not os.path.isdir(retrained_dirout):
        os.mkdir(retrained_dirout)
    imorder = np.linspace(0, n_img_train * len(mark_list_in) - 1, n_img_train * len(mark_list_in), dtype=int)
    otolreader.InternalTransferNetworkTraining.internal_transfer_training.make_trained_transfer_nets(
        pdir, binary_model_prefix, nim_fold, retrain_dir, 'samples_', imorder, retrained_dirout
    )
