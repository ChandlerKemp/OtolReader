OtolReader import InternalTransferNetworkTraining as itnt
import pickle
import os

mark_list = ['3,5H10', '1,6H', 'None', '6,2H', '4n,2n,2H']
itnt.training_array_maker.training_array_generator('../../TrainingSampleDatabase/publication_training_array.p',
                              pdir='../../TrainingSampleDatabase/TrainingSamples', n_img=30, stepsize=10,
                              avg_window=20, ngen=20, inds=None, max_s_per_im=30, mark_list=mark_list)


mark_list = ['None']
itnt.training_array_maker.training_array_generator('../../TrainingSampleDatabase/extra_none_array.p',
                              pdir='../../TrainingSampleDatabase/TrainingSamples', n_img=30, stepsize=10,
                              avg_window=20, ngen=20, inds=None, max_s_per_im=120, mark_list=mark_list)

with open('../../TrainingSampleDatabase/publication_classification_training_array.p', 'rb') as f:
    class_training = pickle.load(f)

with open('../../TrainingSampleDatabase/extra_none_array.p', 'rb') as f:
    bin_training = pickle.load(f)

mark_count = 0
bin_training['mark'] = []
for mark, dat in class_training.items():
    if mark != 'None':
        bin_training['mark'].extend(dat)

with open('../../TrainingSampleDatabase/publication_binary_training_array.p', 'wb') as f:
    pickle.dump(bin_training, f)

btdir = '../../TrainingSampleDatabase/binary_training'
os.mkdir(btdir)

for cl in bin_training:
    os.mkdir(os.path.join(btdir, cl))
    count = 0
    for im in bin_training[cl]:
        with open(os.path.join(btdir, cl, str(count)), 'wb') as f:
            pickle.dump(im, f)
        count += 1

cldir = '../../TrainingSampleDatabase/classification_training'
os.mkdir(cldir)
for cl in class_training:
    os.mkdir(os.path.join(cldir, cl))
    count = 0
    for im in class_training[cl]:
        with open(os.path.join(cldir, cl, str(count)), 'wb') as f:
            pickle.dump(im, f)
        count += 1
