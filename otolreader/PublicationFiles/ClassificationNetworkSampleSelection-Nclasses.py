import pickle
import numpy as np
import multiprocessing
import os
from otolreader.InternalTransferNetworkTraining import classification_network_mark_score_table as cnmst

# ----------------------------------------------Constants--------------------------
mark_list_in = ['3,5H10', '1,6H', 'none', '6,2H', '4n,2n,2H']
n_img = 30  # Number of images per class
n_fold = 10
n_class = 4
if n_fold > 1:
    im_per_fold = n_class * n_img / n_fold  # Number of images in each fold
else:
    im_per_fold = 0
imdir = "../../OtolithImages/"
pdir = os.path.join('..', '..', '..', 'OtolReader-Publication', 'Models', 'N-Class cross val',
                    '{:0.0f} classes'.format(n_class))
im_dir = os.path.join('..', '..', '..', 'Images', 'OtolithImages')
fname_base = os.path.join(pdir, 'classnet_')
cvorder_path = os.path.join(pdir, 'cvorder.p')
with open(cvorder_path, 'rb') as f:
    imorder = pickle.load(f)
mark_list = mark_list_in[:n_class]
ang = np.linspace(-90, 90, 9)
model_in_len = 151
dirout = os.path.join(pdir, 'CrossVal-ClassifierSampleTable-' + str(n_class) + 'class')
shape = [800, 240]
x_step = 120
args = [[fname_base, fold_ind, imorder, im_dir, mark_list, ang, model_in_len, dirout, n_img] for
                fold_ind in range(n_fold)]
# ----------------------------------------Analysis------------------------
if __name__ == '__main__':
    with multiprocessing.Pool(processes=n_fold) as pool:
        pool.starmap(cnmst.find_marks, args)
