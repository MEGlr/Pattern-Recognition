from main_train import *
from evaluate import evaluate
from sklearn.metrics import accuracy_score
from csv import DictWriter

import os.path
import sys

import random


def train_with_random_hyperparameters(dataset_type):
    """
    Trains a model on random hyperparameter values
    for the specified `dataset_type`. This function
    returns the hparams used and the macro f1 score.
    """

    # ================ Look up Tables ===========================

    READ_FN = {
        'spectrogram':      read_mel_spectrogram,
        'chroma':           read_chromagram,
        'fused':            read_fused_spectrogram,
        'spectrogram_beat': read_mel_spectrogram,
        'chroma_beat':      read_chromagram,
        'fused_beat':       read_fused_spectrogram 
    }

    # ================= Pick random hyperparameter values =========

    # choose a value for the hyperparameters
    hparams = {
        # any hyperparameter that is not defined here
        # will take its default value when the train_helper
        # method is called

        # All the available hyperparameters
        # lr
        # batch_train
        # rnn_size
        # rnn_layers
        # bidirectional
        # dropout

        "lr": np.power(10, -np.random.uniform(2, 4.4)),
        "dropout": np.random.uniform(0.1, 0.8),
        # "rnn_size": random.choice([32, 64, 128, 256]),
        "rnn_layers": random.choice([1, 2]),
        # "bidirectional": random.choice([False, False, True])

    }

    # ======================= Train the model =======================

    print("training with hparms:", hparams)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    _, score = train_helper(
            "hyper_tuning",
            LSTMBackbone,
            Classifier,
            dataset_type,
            epochs=50,
            device=device,
            hparams=hparams
        )

    # ============= return the hyperparameters used and the score =========

    return hparams, score

def save_result(csv_filename, hparams, score):
    """
    Save the results to a csv file. Actually this function just appends
    a new line in the `csv_filename` which holds the values of each
    hyperparamter and the score.
    """

    # create the dict which will be saved
    dict_to_write = hparams.copy()
    dict_to_write["score"] = score


    # make the directories if they do not exist
    # its like mkdir -p
    csv_filename = Path(csv_filename)
    csv_filename .parents[0].mkdir(parents=True, exist_ok=True)

    # check if the csv exists
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a') as f_object:

        columns = ["lr","dropout","batch_train","rnn_size","rnn_layers","bidirectional","score"]
     
        dictwriter_object = DictWriter(f_object, fieldnames=columns)

        if not file_exists:
            dictwriter_object.writeheader()
     
        # Pass the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(dict_to_write)
     
        # Close the file object
        f_object.close()



if __name__ == "__main__":

    # Run 20 experiments with different hyperparameters
    for i in range(20):

        print(f"=============== {i+1} ==========")

        # load cli parameter
        if(len(sys.argv) == 1):
            # if the dataset_type is not defined just use 'chroma beat'
            dataset_type = 'chroma_beat'
        else:
            dataset_type = sys.argv[1]

        # train the model with random hyperparameters
        hparams, score = train_with_random_hyperparameters(dataset_type)

        # append the result to the csv file
        save_result(f"../hyper_tuning/lstm_{dataset_type}.csv", hparams, score)

        print(hparams, score)
