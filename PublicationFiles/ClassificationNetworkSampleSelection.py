import pickle
import numpy as np
import multiprocessing
import os
from OtolReader.InternalTransferNetworkTraining import classification_network_mark_score_table as cnmst

# ----------------------------------------------Constants--------------------------
n_img = 30  # Number of images per class
im_per_fold = 15  # Number of images in each fold
n_fold = 10
imdir = "../../OtolithImages/"
pdir = os.path.join('..', '..', 'OtolReader-Publication', 'Models', 'CrossVal-Classifier-5classes-30images')
im_dir = os.path.join('..', '..', 'Images', 'OtolithImages')
fname_base = os.path.join(pdir, 'classnet_')
cvorder_path = os.path.join('..', '..', 'OtolReader-Publication', 'Models',
                            'CrossVal-Classifier-5classes-30Images', 'crossval_order.p')
with open(cvorder_path, 'rb') as f:
    imorder = pickle.load(f)
mark_list = ['3,5H10', '1,6H', 'none', '6,2H', '4n,2n,2H']
ang = np.linspace(-90, 90, 9)
model_in_len = 151
max_im_ind = 150
dirout = os.path.join('..', '..', 'OtolReader-Publication', 'Models', 'CrossVal-ClassifierSampleTable')
shape = [800, 240]
x_step = 120
args = [[fname_base, fold_ind, imorder, im_dir, mark_list, ang, model_in_len, max_im_ind, dirout] for
                fold_ind in range(n_fold)]
# ----------------------------------------Analysis------------------------
if __name__ == '__main__':
    with multiprocessing.Pool(processes=n_fold) as pool:
        pool.starmap(cnmst.find_marks, args)
