import sys

import multiprocessing as mp

from lib import *

from helper_scripts.parser import *
from helper_scripts.plot_confusion_matrix import *

import pickle

N1 = 6
N2 = 2

RUN_IN_PARALLEL=True

# =============== STEP 9 ===============

step_seperator(9)

import git
if( not os.path.exists("./dataset/free-spoken-digit-dataset")):
    git.Git("./dataset/").clone("https://github.com/Jakobovski/free-spoken-digit-dataset.git")


# parse data
X_train, X_test, y_train, y_test, spk_train, spk_test = parser('dataset/free-spoken-digit-dataset/recordings')

# Normalize data using the train set
scale_fn = make_scale_fn(X_train)
X_train = scale_fn(X_train)
X_test = scale_fn(X_test)


# split data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)


#group samples for each digit
X_train_per_digit = [[] for _ in range(10)]


for i in range(len(X_train)):
    X_train_per_digit[y_train[i]].append(X_train[i].tolist())

# =============== STEP 10 and STEP 11 ===============

step_seperator(10)
print("This step is done in conjunction with step 11")
step_seperator(11)

# we train hmms with 4 states and 4 components
hmm_bundle = train_hmm_bundle(X_train_per_digit, 4, 4)

# for i, model in enumerate(hmm_bundle):
#    trans_matrix = model.dense_transition_matrix() 
#    print("======================")
#    print(i)
#    print(trans_matrix)


acc = hmm_bundle_accuracy(hmm_bundle, X_val, y_val) 
print(f"The accuracy is {acc * 100 : .2f} %")

# =============== STEP 12 ===============

step_seperator(12)

# help function in order to run it in parallel
def train_hmm_bundle_helper(hparams):
    """
    Trains a hmm_bundle using the given hparams.

    Args:
        hpams = (num_states, num_components)
        where
            num_states : the number of states in the hmms
            num_components : number of components for the gmm

    Returns: [hmm_bundle, accuracy]
        where
            hmm_bundle : the hmm bundle that was trained
            accuracy : the accuracy of the model on the validation set
    """

    # unravel the hyper parameters
    num_states, num_components = hparams

    # train the model
    hmm_bundle = train_hmm_bundle(X_train_per_digit, num_states, num_components, False)

    # calculate the accuracy
    accuracy = hmm_bundle_accuracy(hmm_bundle, X_val, y_val)

    # return the results
    return [hmm_bundle, accuracy]


# bundles_and_accuracies[i][j] = [accuracy, hmm_bundle]
# with hmm_bundle being a bundle of hmms using i states
# and gmms of j components
bundles_and_accuracies = [
        [
            [-2,"init"]
            for num_components in range(0,6)
        ]
        for num_states in range(0,5)
    ]

# This takes a lot of time.
# Try and load the pickle from the
# previous run. If it does not exist
# then compute it

try:
    bundles_and_accuracies = pickle.load(open("bundles_and_accuracies.pickle", "rb"))
    print("Loaded pickle!")
except (OSError, IOError) as e:
    
    
    print("Now computing bundles_and_accuracies")

    if RUN_IN_PARALLEL == True:

        # Here we use the multiprocessing package of python
        # to train lots of hmm bundles. The code below is
        # equivalent to the two nested loops that can be seen
        # in the else case.

        with mp.get_context('fork').Pool(mp.cpu_count()-1) as pool:

            # the list of hparams to train for
            hparams = []

            for num_states in range(1,5):
                for num_components in range(1,6):
                    hparams.append((num_states, num_components))
                    
            # train models in parallel
            tmp = list(tqdm(pool.imap(
                    train_hmm_bundle_helper,
                    hparams
                ), file=sys.stdout, total=len(hparams)))

            # reshape tmp into the bundles_and_accuracies 2D list
            for num_states in range(1,5):
                for num_components in range(1,6):
                        bundles_and_accuracies[num_states][num_components] = tmp[(num_states-1) * 5 + (num_components-1)]

    else:
        # run sequentially

        for num_states in tqdm(range(1, 5), file=sys.stdout): # states 1-> 4
            for num_components in tqdm(range(1, 6), file=sys.stdout, leave=False): # components 1-> 5

                bundles_and_accuracies[num_states][num_components] = train_hmm_bundle_helper((num_states, num_components))

    # save the models
    pickle.dump(bundles_and_accuracies, open("bundles_and_accuracies.pickle", "wb"))


# unpack bundles and accuracies in two different 2D lists

# this 2D list contains only the models/hmm bundles
bundles = [
        [
            bundles_and_accuracies[num_states][num_components][0] 
            for  num_components in range(1, 6)
        ]
        for num_states in range(1, 5)
    ] 

# this 2D list contains only the accuracies
accuracies = [
        [
            bundles_and_accuracies[num_states][num_components][1] 
            for  num_components in range(1, 6)
        ]
        for num_states in range(1, 5)
    ] 
accuracies = np.array(accuracies)


# Find the model with the highest accuracy
best_num_states, best_num_components = np.unravel_index(accuracies.argmax(), accuracies.shape)
best_accuracy = accuracies[best_num_states][best_num_components]

best_model = bundles[best_num_states][best_num_components]

best_test_accuracy = hmm_bundle_accuracy(best_model, X_test, y_test)
            
print(f"Best accuracy on validation set = {best_accuracy:.2f},\n using {best_num_states+1} states and {best_num_components+1} components.")
print(f"Using these # states and  # components, accuracy on test set is {best_test_accuracy * 100 : .2f} %")        

# plot accuracies on heatmap
plot_hparams_acc(accuracies,
        title="Accuracy for different hparams",
        filename="figures/step_12_accuracies.png")

# =============== STEP 13 ===============

step_seperator(13)

# make the confusion matrix on the validation set
cm  = hmm_bundle_cm(best_model, X_val, y_val)
plt.figure()
plot_confusion_matrix(cm, [f"digit {i}" for i in range(10)], title='Confusion matrix on Validation Set')
magic_plt_show("figures/step_13_cm_val.png")

# make the confusion matrix on the test set
cm = hmm_bundle_cm(best_model, X_test, y_test)
plt.figure()
plot_confusion_matrix(cm, [f"digit {i}" for i in range(10)], title='Confusion matrix on Test Set')
magic_plt_show("figures/step_13_cm_test.png")


