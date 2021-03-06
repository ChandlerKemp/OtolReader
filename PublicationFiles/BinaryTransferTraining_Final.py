import os
import pickle
import numpy as np
import sys
sys.path.append('../../')
import otolreader as otolr

basedir = '../../OtolReader-Publication/Models'
with open(os.path.join(basedir, 'crossval_order.p'), 'rb') as f:
    cvorder = np.array(pickle.load(f))
pdir = os.path.join(basedir, 'CrossVal-Binary-5classes-30Images')
model_prefix = 'binarynet_'
nim_fold = 15
sampledir = os.path.join(basedir, 'CrossVal-ClassifierSampleTable')
samps_prefix = 'ClassifierNetworkForSelection_SmoothedSamples_'
dirout = os.path.join(basedir, 'Retrained-Binary-Nets')
# os.mkdir(dirout)

otolr.InternalTransferNetworkTraining.internal_transfer_training.make_trained_transfer_nets(
    pdir, model_prefix, nim_fold, sampledir, samps_prefix, cvorder, dirout
)

