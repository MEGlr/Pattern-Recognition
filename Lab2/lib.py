import os
import re
import random
from glob import glob
from pathlib import Path

from tqdm import tqdm

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix

import statistics as stats
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import norm

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


from pomegranate import *

from helper_scripts.plot_confusion_matrix import *

import pickle


# =============== MISC =============== 

def step_seperator(step_num):
    """
    Just prints a message to the stdout
    to inform the user which step is currently running
    """

    print("\n\033[1;32m[*] Running Step {} ...\033[0m\n".format(step_num))


def magic_plt_show(filename=""):
    """
    if filename is not empty then it saves the plot.
    if filename is empty then the plot is just shown
    """

    if filename:
        # save the plot with the specified filename

        path = Path(filename)

        # make the directories if they do not exist
        # its like mkdir -p
        path.parents[0].mkdir(parents=True, exist_ok=True)

        print("Saving figure at:", path)
        plt.savefig(path, dpi=300);
    else:
        # just show the plot
        plt.show()


# =============== STEP 2 =============== 

def data_parser(directory="dataset/digits"):
    """
    Reads all the wav files and returns 3 lists and the sampling rate.
    The first list containts the audio from the wav files.
    The second list contains the id of the speaker.
    The third list contains which digit was told.
    """

    files = glob(os.path.join(directory, "*.wav"))
    
    # get only the filenames without the path and the .wav
     
    fnames = [os.path.basename(path) for path in files]
    fnames = [f.split(".")[0] for f in fnames]

    # get the sampling rate of the audio
    _, sampling_rate = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)
        return wav

    def parse_digit_labels(text):
        LUT = {
            "one":      1,
            "two":      2,
            "three":    3,
            "four":     4,
            "five":     5,
            "six":      6,
            "seven":    7,
            "eight":    8,
            "nine":     9
        }

        return LUT[text]


    # a regular expression
    # it is used to extract the 
    # digit and the speaker id
    regex = re.compile("([a-z]*)([0-9]*)")

    # the three lists to return
    wavs = [read_wav(f) for f in files]
    speaker_ids = [regex.search(f).group(2) for f in fnames]
    digits = [regex.search(f).group(1) for f in fnames]

    # parse text to integers
    digits = [parse_digit_labels(x) for x in digits]

    return wavs, speaker_ids, digits, sampling_rate


# =============== STEP 3 =============== 

def extract_features(wavs, n_mfcc, Fs):
    # Extract MFCCs for all wavs
    window = 25 * Fs // 1000 # 25 msec
    step = 10 * Fs // 1000 # 10 msec

    def all_features(wav):
        mfcc = librosa.feature.mfcc(
                y=wav,
                sr=Fs,
                n_fft=window,
                hop_length=step,
                n_mfcc=n_mfcc
            )

        mfcc_delta =  librosa.feature.delta(mfcc)
        mfcc_delta_delta =  librosa.feature.delta(mfcc, order=2)

        # concatenate all the features together
        # the final shape is (#frames, n_mfcc * 3)
        result = np.concatenate((mfcc.T, mfcc_delta.T, mfcc_delta_delta.T), axis=1)

        return result
            

    features = [
        all_features(wav)
        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame and delta and delta-delta".format(n_mfcc))

    return features

# =============== STEP 4 =============== 

def plot_hist_step_4(mfccs, labels, for_digit, plot_title, plot_filename):
    """

    Makes the histogram for step 4.

    mfccs : the list that contains all the mfccs of the dataset
    labels: a list of which digit was told in each utternace
    for_digit: for which digit to create the histogram

    plot_title: the title of the plot
    plot_filename: the filename of the plot when it is going to be saved
    """

    def help_create_subplot(which_mfcc):

        print(which_mfcc)

        # an array that holds all the values that will be used for
        # the histogram
        selected_values = []

        for i, label in enumerate(labels):

            # if the utterance is of the requested digit
            if (label == for_digit):
                # get the selected mfcc for all the frames of this utternace
                selected_values += list(mfccs[i][:, which_mfcc])

        std = np.std(selected_values)

        # create the histogram
        plt.hist(selected_values, bins=50)

        plt.title("Histogram for the {} mfcc".format(
            "1st" if (which_mfcc == 0) else "2nd"
            ))

        plt.xlabel("Values of the {} mfcc".format(
            "1st" if (which_mfcc == 0) else "2nd"
            ))

        plt.ylabel("Count")

    # start creating the plot
    plt.figure()

    # create subplot for the first mfcc
    plt.subplot(1,2,1)
    help_create_subplot(0) # counting from 0

    # create subplot for the second mfcc
    plt.subplot(1,2,2)
    help_create_subplot(1)

    plt.suptitle(plot_title)

    plt.tight_layout()
    magic_plt_show(plot_filename)



# =============== STEP 4b =============== 


def extract_mfsc(wavs, n_mfsc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.melspectrogram(
            y=wav, sr=Fs, n_fft=window, hop_length=window - step, n_mels=n_mfsc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfsc features...")
    ]

    print("Feature extraction completed with {} mfscs per frame".format(n_mfsc))

    return frames


def help_find_indexes_2spk(for_digits, labels):
    """
    for_digits: the list of digits for which we are interested
    labels: a list of which digit was told in each utternace
    returns a list of the indexes of two different speakers pronouncing each of the digits in for_digits
    """
    indexes = []
    labels = np.array(labels)
    for digit in for_digits:
        #firtst index that corresponds to the digit in question
        digit_ind = np.where(labels == digit)[0]
        #we want 2 speakers so we select 2 indexes (each one will correspond to a different speaker) (since they all .wavs for the same digit are grouped together - indexes that correspond to them are continuous)
        indexes.append(digit_ind[0])
        indexes.append(digit_ind[1])
    return indexes

                
def plot_cor( mf, labels, for_digits, plot_title, plot_filename):

    """
    mf: the list that contains all the mfccs or mfscs of the dataset
    labels: a list of which digit was told in each utternace
    for_digit: a list of digits the correlation of their
    mfccs/mfscs we wish to plot
    plot_title: the title of the plot
    plot_filename: the filename of the plot when it is going to be saved
    which_wavs = help_find_wavs_2spk(for_digits, labels, wavs)
    """
    ind_of_wav_for_digits = help_find_indexes_2spk(for_digits, labels)
    # an array that holds all the MFSCSs or MFSCSs values that will be used 
    values = []
    for i in ind_of_wav_for_digits:
        values.append(list(mf[i][:,:13]))
    print(len(values))
    fig, ax = plt.subplots(1, 2*int(len(for_digits)), figsize=(15,5))
    for i, mf_ in enumerate(values):
        # Fs=8000
        # window = 25 * Fs // 1000 # 25 msec
        # step = 10 * Fs // 1000 # 10 msec
        # mf_ =  librosa.feature.melspectrogram(
        #     wavs[ind_of_wav_for_digits[0]], Fs, n_fft=window, hop_length=window - step, n_mels=13
        # ).T

        #corr() is a function available only for DataFrames => cast array to Panda DataFrame
        panda_frame = pd.DataFrame.from_records(mf_)
        ax[i].imshow(panda_frame.corr())
        ax[i].title.set_text(f'Correlation of {plot_title}\n for Speaker {i%2+1}, Digit {for_digits[i%2]}')
    plt.tight_layout()
    magic_plt_show(plot_filename)

# =============== STEP 5,6  ===============


# plotting 2/3 dimsensions of input vectors
def scatter(X, y, plot_title, plot_filename, dim=2):
    colors = list(mcolors.TABLEAU_COLORS.values())
    markers = ["o", "v", "^", "<", ">", "P", "8", "s", "h"]
    y = np.array(y)
    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot()
        #keep only the 1rst 2 dimentions
        X0, X1 = X[:, 0], X[:, 1]
        for i in range(1, 10):
            ax.scatter(X0[y == i], X1[y == i], 
                    c=colors[i], marker = markers[i-1],
                    label=i, s=80, alpha=1, edgecolors=colors[i])
            
    if dim == 3: 
        ax = fig.add_subplot(projection='3d')
        X0, X1, X2 = X[:, 0], X[:, 1], X[:, 2]
        for i in range(1, 10):
            ax.scatter(X0[y == i], X1[y == i], X2[y == i],
                    c=colors[i], marker = markers[i-1],
                    label=i, alpha=1, edgecolors=colors[i])
            
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()
    ax.title.set_text(plot_title)
    magic_plt_show(plot_filename)
    


# =============== STEP 7 ===============

# =============== Custom NB Classifier ===============


def digit_mean(X, y, digit):
    """Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    """
    y = np.array(y)
    digit_indices = np.where(y == digit)[0]
    print
    if (len(digit_indices) == 0):
        # if there are no samples in X for the class `digit`
        # return a 16x16 matrix with elements of value INF

        # We choose to return an "INF" matrix and not a
        # zero one so that the euclidean classifier
        # never classifies a sample to a category that is
        # not present in the train set.

        INF = 1e10
        means = np.full(len(X[0]), INF)
    else:
        # if there are samples in X for the class `digit`
        # then just calculate the mean on each pixel.
        X = X[y==digit]
        #print(np.array(X).shape)
        means = [
            np.mean(X, axis = 0)
        ]
    #print(np.array(means).shape)
    # return the mean values as a vector
    return np.array(means)


def digit_variance(X, y, digit):
    """Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    """
    y = np.array(y)
    X = X[y==digit]
    # calculate the variance for every feature for utterances of the digit given in input
    variances = [
        np.var(X, axis = 0)
    ]

    # return the variance values as a vector
    return np.array(variances)
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    y = np.array(y)
    # calcluate how many samples are in each class
    # counts[i] = number of samples for class i
    counts = np.bincount(y)
    #ingore index = 0 (as it corresponds to digit 0 we do not have in our dataset - this is a technicality as we want to use bincount function)
    counts = counts[1:]
    return counts / len(y)

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        ########################################################
        # Calculate the mean value for each pixel for each class
        ########################################################

        self.X_mean_ = np.zeros((9, len(X[0])))

        # compute the mean for each digit
        for i in range(1, 10):
            self.X_mean_[i-1] =  digit_mean(X, y, i)


        ####################################################
        # Set the variance for each pixel and for each class
        ####################################################

        # by default use unit variance
        self.X_variance_ = np.zeros((9, len(X[0])))

        if not self.use_unit_variance:

            # if self.use_unit_variance == False
            # calculate the variance

            for i in range(1, 10):
                self.X_variance_[i-1] = digit_variance(X, y, i)

            # If the variance is zero for some pixel
            # then later a divide by zero exception is thrown.
            # So we add a small value to every variance of each pixel
            self.X_variance_ += 1e-5

        ######################################
        # Calculate the a-priori probabilities
        ######################################

        self.priors = calculate_priors(X, y) 


        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """

        # we want to calculate argmax_{y=0,..,9} P(x|y) * P(y)
        # for each sample in X

        # predictions[i] = prediction for X[i]
        predictions = []

        for sample in X:

            # direct[i] = P(x=sample|y=i)
            direct = [self.direct_prob(sample, y) for y in range(1, 10)]
            direct = np.array(direct)

            # the prediction for this sample
            pred = np.argmax(direct * self.priors)+1 #we add 1 because indexes of the array are from 0 to 9 and correspond to digits 1 to 10
            
            predictions.append(pred)


        return predictions

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """

        predictions = np.array(self.predict(X))
        y = np.array(y)
        accuracy = (np.sum(predictions == y))/len(y)

        return accuracy

    def direct_prob(self, x, y):
        """
        Calculates the direct probability. a.k.a. P(x|y)
        with x being the image and y the class 

        x is expected to be a numpy vector of length 256
        y is expected to be an integer from 0 up to 9
        """

        # all pixels values are assumed to be independent
        # so P(x|y) = P(x_0|y) * P(x_1|y) * ... * P(x_255|y)
        # where x_i are the pixel values

        # p_pixel[i] = P(x_i | y)
        p_pixel = norm.pdf(
                x,
                loc=self.X_mean_[y-1],
                scale=np.sqrt(self.X_variance_[y-1])
            )

        # and multiply all of them together
        return np.prod(p_pixel)


# =============== STEP 8 ===============

def gen_data_step_8(N=200):
    """
    This function generates sequences of 10 samples
    from the sin and cos funtion (f=40Hz). It generates
    lots of these sequences in order to train a recursive model. 

    Args:
        N : the number of sin and cos pairs/sequences to generate
    Returns:
        list of sin and cos sequences
    """

    # 320Hz sampling rate
    # corresponds to 8 samples per period
    Fs = 320
    Ts = 1/ Fs

    # frequency and period of functions
    freq = 40
    T = 1/ freq

    # the number of sample to take from the
    # sin and the cos for each training sequence
    samples_per_sequence = 10

    # The two lists that will hold all the
    # training examples-sequences.
    # x will hold the sin sequences.
    # y will hold the cos sequences.
    x = []
    y = []

    for _ in range(N):

        # pick a random stating time
        start_time = random.uniform(0, T)

        # get the moments in time where we are going to
        # calculate the sin and cos function value
        t = start_time + Ts * np.arange(samples_per_sequence)

        # get the sin and cos sequence
        x_samples = np.sin( 2 * np.pi * freq * t)
        y_samples = np.cos( 2 * np.pi * freq * t)

        # append them to the list
        x.append(x_samples)
        y.append(y_samples)

    # return the training samples that are generated
    return np.array(x), np.array(y)


class RecursiveCosPredictor(nn.Module):

    def __init__(self, recursive_layer, hidden_dim, seq_len):
        """
        Args:
            recursive_layer: is a str and takes the values
                "RNN", "LSTM", "GRU". This parameter is
                used to select the recursive layer.

            hidden_dim: the size of the hidden dimension.

            seq_len: the length of the sequnce. In our case
                it is 10, because the model takes 10 samples
                as its input.
        """

        super(RecursiveCosPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # select the recursive layer
        if (recursive_layer == "RNN"):
            self.rec_layer = nn.RNN(1, hidden_dim, batch_first=True)
        elif(recursive_layer == "LSTM"):
            self.rec_layer = nn.LSTM(1, hidden_dim, batch_first=True)
        elif(recursive_layer == "GRU"):
            self.rec_layer = nn.GRU(1, hidden_dim, batch_first=True)

        # we use a fully connected layer to make our
        # prediction from the hidden state
        self.fc = nn.Linear(hidden_dim, 1)
        

    def forward(self, x):

        # the shape of x is (batch_size, seq_len)
        # we need to reshape it to (batch_size, seq_len, 1)
        x = torch.reshape(x, (-1, self.seq_len, 1))



        # the initial hidden state is 0
        # which is the default if it is
        # not defined

        # hidden_states has shape (batch_size, seq_len=10, hidden_dim)
        hidden_states, _ = self.rec_layer(x)


        # apply the fully connected layer on the hidden states
        out = self.fc(hidden_states)

        # reshape back to (batch_size, seq_len)
        out = torch.reshape(out, (-1,self.seq_len))

        return out

def eval_regression_model_on_dataset(model, dataloader):
    """
    Evaluates the model on the given dataloader
    and returns the predictions and the ground truth values.

    It returns a tuple (y_true, y_pred)

    where
        y_true: contains the floating point values which are
            extracted from the dataloader (ground truth)
        y_pred: are the predictions of the model
            on the given dataset
    """

    y_pred = []
    y_true = []

    with torch.no_grad():

        for inputs, ground_truth in dataloader:

            output = model(inputs)

            # append the predictions and the ground truth values
            y_pred += output.tolist()
            y_true += ground_truth.tolist()

    return y_true, y_pred


def train_regression_model(x, y, split_perc, model, criterion, optimizer, val_criterion, batch_size, epochs):
    """
    This function trains a regression model.

    Args:

        x: a list of the inputs of all the sample

        y: the target value for each sample

        split_perc: the train/validation set split. e.g. if it is 0.8
            then 80% of the given dataset is used for training
            and the rest is used for validation

        model: a pytorch model

        criterion: the training criterion/loss

        optimizer: the optimizer to use (e.g Adam)

        val_criterion: the critirion used on the validation setp (e.g. MSE)

        batch_size: the batch size

        epochs: the number of epochs

    Returns: (train_hist, val_hist), (val_x, gt_y pred_y)

        (train_hist, val_hist): The train and the validation score history.
            For example the MSE computed on the test set and the validation 
            set for each epoch.

        (val_x, gt_y pred_y):
            val_x: a list of all the input values from the validation set
            gt_y: the ground truth values taken from the validation set
            pred_y: the predictions of the model on the validation set.
                The predictions are made on the last epoch.
    """


    X_train, X_val, Y_train, Y_val = train_test_split(
        x, y,
        train_size=split_perc, shuffle=True, random_state=42
    )

    # Create PyTorch dataset
    train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float),
            torch.tensor(Y_train, dtype=torch.float)
        )

    val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float),
            torch.tensor(Y_val, dtype=torch.float)
        )

    # Create the initializers
    train_loader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True
        )

    # do not shuffle the validation set
    # there is no need. Also if we shuffle
    # it when we return the (val_x, gt_y pred_y)
    # they will not match to each other
    val_loader = DataLoader(
            val_dataset,
            batch_size = batch_size,
        )

    # lists that hold the score(e.g. MSE) on
    # the train and test set for each epoch
    train_hist = []
    val_hist = []

    # The predictions of the model on the validation set.
    # These predictions are made after the last epoch.
    pred_y = []

    # Train the model
    for epoch in range(epochs):

        # set model to train mode, not needed in our case but ok
        model.train()

        # train on the whole train set
        for inputs, ground_truth in train_loader:

            # reset the accumulated gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate loss
            loss = criterion(outputs, ground_truth)

            # compute the gradients
            loss.backward()

            # update the model parameters
            optimizer.step()

        # set model to evaluation mode, not needed
        model.eval()

        # calculate the val_criterion on the train set
        y_true_train, y_pred_train = eval_regression_model_on_dataset(model, train_loader)
        train_score = val_criterion(y_true_train, y_pred_train)

        # calculate the val_criterion on the validation step
        y_true_val, y_pred_val = eval_regression_model_on_dataset(model, val_loader)
        val_score = val_criterion(y_true_val, y_pred_val)

        # push scores to the history
        train_hist.append(train_score)
        val_hist.append(val_score)

        # if it is the last epoch save
        # the prediction on the validation set
        # in order to return it
        if(epoch == epochs - 1):
            pred_y = y_pred_val

        print(f"Epoch {epoch: <3}: train score = {train_score:.3f}, validation score = {val_score:.3f}")

    return (train_hist, val_hist), (X_val, Y_val, pred_y)



def plot_leaning_curves(train_loss, val_loss, title, filename):
    """
        plots the learning curves
    """

    plt.figure()

    plt.title(title)

    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="validation loss")
    plt.legend()

    plt.xlabel("epoch")
    plt.ylabel("MSE")

    plt.tight_layout()
    magic_plt_show(filename)


def subplot_example(x, ground_truth, prediction, title):
    """
        plot subplot
    """

    plt.plot(x, label="input")
    plt.plot(ground_truth, label="ground_truth")
    plt.plot(prediction, label="prediction")

    plt.legend()

    plt.title(title)




# =============== STEP 10 ===============

# pomegranate is unstable. We create a custom exception
# in case it does not train the hmm model
class DidNotTrainError(Exception):
    pass

def get_diag_gaussian(dim):
    list_of_ind_gaussians = []
    for _ in range(dim):
        mu = np.random.normal(loc=0.0, scale=1.0, size=1)
        list_of_ind_gaussians.append( NormalDistribution(mu, 1) )
    
    return IndependentComponentsDistribution(list_of_ind_gaussians)

def get_set_of_diag_gaussians(n, dim):
    gaussians = []
    for _ in range(n):
        gaussians.append(get_diag_gaussian(dim))

    return gaussians

def hmm(data, n_states, n_components, verbose=True):
    """
    This function trains a single hmm on the given data.

    data : all the data for a digit (e.g all the sample that say "one") (can be a numpy array)
    n_states  : the number of HMM states
    n_components : the number of Gaussians
    """
    
    
    

    dists = [] # list of probability distributions for the HMM states
    for i in range(n_states):
        if n_components > 1:
            gaussians_list = get_set_of_diag_gaussians(n_components, len(data[0][0]))
            a = GeneralMixtureModel(gaussians_list)
        else:
            a = MultivariateGaussianDistribution.from_samples(np.array(data[0]), inertia=0.8)
        dists.append(a)

    trans_mat = np.zeros((n_states,n_states)) # your transition matrix
    for i in range(n_states):
            for j in range(n_states):
                if i == j or j == i+1:
                    trans_mat[i,j] = 0.5 
                    # από κάθε κατάσταση υπάρχουν 2 μεταβάσεις, τις οποίες
                    # θεωρούμε ισοπίθανες 


    starts = np.zeros(n_states) # your starting probability matrix
    starts[0] = 1 #π𝑖 = {0 για 𝑖 ≠ 1 και 1 για 𝑖 = 1}

    ends = np.zeros(n_states) # your ending probability matrix
    ends[-1] = 0.5 # only the last state can transition to the "end" state


    # Define the GMM-HMM
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])

    tran_matrix_before_fit = model.dense_transition_matrix()

    # Fit the model
    #model.fit(data, algorithm='baum-welch', inertia=0.8, verbose=verbose, stop_threshold=100)
    model.fit(data, algorithm='baum-welch', inertia=0.8, verbose=verbose, max_iterations=15)

    tran_matrix_after = model.dense_transition_matrix()


    # Sometimes after the fitting the transition matrix does
    # not change. If that is the case we through an exception
    if(np.linalg.norm(tran_matrix_after - tran_matrix_before_fit) < 1e-5):
        # the transition matrix probably did not change
        raise DidNotTrainError() 


    return model

def safe_hmm(data, n_states, n_components, verbose):
    """
    Pomegranate is numerically astable. If if run it
    multiple times at some point it runs without
    exceptions. So this function calls the `hmm`
    function until it does not through any exceptions.
    """
    
    # call hmm until it throws no exceptions
    while True:
        try:
            _hmm = hmm(data, n_states, n_components, verbose=verbose)
            return _hmm 
        except DidNotTrainError as e:
            print("Did not train. Retrying...")
            pass
        except np.core._exceptions._UFuncOutputCastingError as e:
            print(".", end="")
            pass



def train_hmm_bundle(data_per_digit, n_states, n_components, debug_print=True):
    """
    This function trains a bundle/set of hmms.
    It trains one hmm for each digit with the
    given number of states and the given number
    of components for the gmm.

    data_per_digit: all the data for all the digits, grouped by the digit
    n_states: the number of states in the hmm
    n_components: the number of components each gmm has
    """

    # digit_hmm[i] = the hmm model for the digit i
    digit_hmm = []

    # train the hmms
    for digit in range(10):
        if(debug_print):
            print(f"Training hmm for digit {digit}")

        # train the hmm using the safe_hmm function
        hmm = safe_hmm(
                data_per_digit[digit], 
                n_states, n_components,
                verbose=debug_print
            )

        # append it to the list
        digit_hmm.append(hmm)

    return digit_hmm

# =============== STEP 12 ===============


def hmm_bundle_predict(hmm_bundle, X):
    """
    Classifies the samples of X using the
    givn hmm_bundle.
    
    Args:
        hmm_bundle: the list of all the hmm models
        X: the data to classify
    """

    # if the samples are as numpy arrays
    # make them python lists. Otherwise
    # pomegranate gives other results
    if(type(X[0]) == np.ndarray):
        X = [x.tolist() for x in X]

    # pred will hold the classes for each sample
    # of X at the end
    pred = np.zeros(len(X))

    for i in range(len(X)):

        # We need the log_likelihood for each 
        # hmm for the current sample X[i].
        log_likelihood = np.zeros(len(hmm_bundle))
        
        # got through the models to calculate the log prob
        for j, model in enumerate(hmm_bundle): 
            logp, _ = model.viterbi(X[i]) # Run viterbi algorithm and return log-probability
            log_likelihood[j] = logp
                
        # classify to the class of the
        # hmm with the maxium log prob 
        pred[i] = np.argmax(log_likelihood)

    return pred

def hmm_bundle_accuracy(hmm_bundle, X, y):
    """
    Calculates the accuracy for the given
    hmm_bundle and returns it.

    Args:
        hmm_bundle: the list of all the hmm models
        X: set of samples to calculate the accuracy on
        y: labels for every sample in X
        returns accuracy of the model
    """

    pred = hmm_bundle_predict(hmm_bundle, X)

    accuracy = accuracy_score(y, pred)

    return accuracy


def plot_hparams_acc(accuracies, title, filename):
    """
    This function prints and plots the accuracies
    for different hyper parameters.
    """

    plt.figure()

    plt.imshow(accuracies, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.yticks(np.arange(accuracies.shape[0]), np.arange(accuracies.shape[0]) + 1)
    plt.xticks(np.arange(accuracies.shape[1]), np.arange(accuracies.shape[1]) + 1)

    fmt = '.2f'
    thresh = (accuracies.max() - accuracies.min()) / 2. +  accuracies.min()
    for i, j in itertools.product(range(accuracies.shape[0]), range(accuracies.shape[1])):
        plt.text(j, i, format(accuracies[i, j] * 100, fmt),
                 horizontalalignment="center",
                 color="white" if accuracies[i, j] > thresh else "black")

    plt.ylabel('Number of States')
    plt.xlabel('Number of Components')
    plt.tight_layout()

    magic_plt_show(filename)

# =============== STEP 13 ===============

def hmm_bundle_cm(hmm_bundle, X, y):
    """
    Calculates the confusion matrix for the given
    hmm_bundle and returns it.

    Args:
        hmm_bundle: the list of all the hmm models
        X: set of samples to calculate the confusion matrix on
        y: labels for every sample in X
    """

    pred = hmm_bundle_predict(hmm_bundle, X)
    
    cm = confusion_matrix(y, pred)

    return cm


# ============================= STEP 14 ===========================

def eval_lstm_model_on_dataset_logits(model, dataloader):
    """
    Evaluates the model on the given dataloader
    and returns the output(the logits) and the ground truth labels.

    It returns a tuple (labels, outputs)

    where
        labels: (a pytorch tensor) contains the labels which are
            extracted from the dataloader (ground truth)
        outputs: (a pytorch tensor) are the outputs of the model
            on the given dataset
    """

    # labels_for_batch[i] = pytorch tensor that contains the labels
    # for each sample in the minibatch i
    # the same is true for outputs_for_batch
    labels_for_batch = []
    outputs_for_batch = []

    with torch.no_grad():

        for inputs, labels, lengths in dataloader:
            
            output = model(inputs, lengths)

            # append the outputs and the labels
            outputs_for_batch.append( output )
            labels_for_batch.append( labels )

    # concat the pytorch tensors one on top of the other
    return torch.cat(labels_for_batch), torch.cat(outputs_for_batch)


def eval_lstm_model_on_dataset(model, dataloader):
    """
    Evaluates the model on the given dataloader
    and returns the predictions (the class) and the ground truth labels.

    It returns a tuple (y_true, y_pred)

    where
        y_true: contains the labels which are
            extracted from the dataloader (ground truth)
        y_pred: are the predicted classes for each sample of the
            given dataset
    """


    labels, outputs = eval_lstm_model_on_dataset_logits(model, dataloader)
    pred = torch.argmax(outputs, dim = 1)

    return labels.tolist(),  pred.tolist()


def train_LSTM_model(model, train_dataloader, val_dataloader, optimizer, epochs, model_name = "simple_lstm", patience = 10, early_stopping=False):
    """
    This function trains the model (LSTM).

    Args:

        model: a pytorch model

        train_dataloader: the dataloader for the training set

        val_dataloader: the dataloader for the validation set

        split_perc: the train/validation set split. e.g. if it is 0.8
            then 80% of the given dataset is used for training
            and the rest is used for validation

        optimizer: the optimizer to use (e.g Adam)

        epochs: the number of epochs

    Returns: (train_hist_acc, train_hist_acc, train_hist_loss, val_hist_loss)

        train_hist_acc, val_hist_acc: The accuracy on the train and the validation set
            for every epoch.
        train_hist_loss, val_hist_loss: The cross-entropy loss on the train and the validation set
            for every epoch.
    """

    print(f"\n\nTraining {model_name} model ...\n")

    # lists that hold the accuracy on
    # the train and val set for each epoch
    train_hist_acc = []
    val_hist_acc = []

    # the same but for the loss
    train_hist_loss = []
    val_hist_loss = []


    criterion = nn.CrossEntropyLoss()
    
    if early_stopping:
        #initialize best score value
        best_loss = 1e10

        # counter counts how many epochs have not improved the val loss 
        counter = 0

        #is it time to stop? 
        early_stop = False 
    
    # Train the model
    for epoch in range(epochs):

        # #initialize epoch loss
        # epoch_loss = 0

        # set model to train mode, not needed in our case but ok
        model.train()

        # train on the whole train set
        for inputs, labels, lengths in train_dataloader:

            # reset the accumulated gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs, lengths)

            # calculate loss
            loss = criterion(outputs, labels)

            # compute the gradients
            loss.backward()

            # update the model parameters
            optimizer.step()
            
            # #update total loss for this epoch 
            # epoch_loss += loss.item()  #use .item() to extract loss value as python float


        # set model to evaluation mode, not needed
        model.eval()

        # calculate the accuracy on the train set
        y_true_train, y_pred_train = eval_lstm_model_on_dataset(model, train_dataloader)
        train_score = accuracy_score(y_true_train, y_pred_train)

        # calculate the accuracy on the validation step
        y_true_val, y_pred_val = eval_lstm_model_on_dataset(model, val_dataloader)
        val_score = accuracy_score(y_true_val, y_pred_val)

        # calculate the cross-entropy loss on the train set
        labels, outputs = eval_lstm_model_on_dataset_logits(model, train_dataloader)
        train_loss = criterion(outputs, labels)

        # calculate the cross-entropy loss on the validation step
        labels, outputs = eval_lstm_model_on_dataset_logits(model, val_dataloader)
        val_loss = criterion(outputs, labels)


        if early_stopping:
            if val_loss > best_loss:
                # if the loss is worst than the best
                counter +=1
                if counter >= patience:
                    early_stop = True   
            else:
                # we reached a new minimum for the val_loss

                #update best loss
                best_loss = val_loss

                #reset counter
                counter = 0

                #save best model
                #torch.save(model, "./checkpoints/" + model_name + "_best")
                fpath = Path("./checkpoints/" + model_name + "_best" + ".pickle")
                pickle.dump(model, open(fpath, "wb"))
                
                
        # make a checkpoint every 5 epochs
        if(epoch%5 == 0):
            #torch.save(model, "./checkpoints/" + model_name + f"_epoch_{epoch}")
            fpath = Path("./checkpoints/" + model_name + f"_epoch_{epoch}" + ".pickle")
            pickle.dump(model, open(fpath, "wb"))
                   
        # push scores to the history
        train_hist_acc.append(train_score)
        val_hist_acc.append(val_score)
        train_hist_loss.append(train_loss)
        val_hist_loss.append(val_loss)

        
        print(f"Epoch {epoch: <3}: train score = {train_score:.3f}, validation score = {val_score:.3f} \n" + 
              f"           train loss = {train_loss:.3f}, validation loss = {val_loss:.3f}.")
        
        #if early_stopping is used in training 
        # and it is time to finish return

        if (early_stopping and early_stop):
            print("\nTraining terminated due to early stopping.")
            return (train_hist_acc, val_hist_acc, train_hist_loss, val_hist_loss)

    if (not early_stopping):
        # torch.save(model, "./checkpoints/" + model_name + "_last_epoch")
        fpath = Path("./checkpoints/" + model_name + "_last_epoch" + ".pickle")
        pickle.dump(model, open(fpath, "wb"))
        
    return (train_hist_acc, val_hist_acc, train_hist_loss, val_hist_loss)


def plot_leaning_curves_lstm(train_acc, val_acc, train_loss, val_loss, title, filename):
    """
        Plots the learning curves.
        It plots the accuracies and losses
        on the train and validation sets
        on the same plot.
    """

    fig, ax1 = plt.subplots()

    ax1.set_title(title)

    ax2 = ax1.twinx()

    line1 = ax1.plot(train_loss, label="train loss")
    line2 = ax1.plot(val_loss, label="val loss")
    line3 = ax2.plot(train_acc, ':', label="train accuracy")
    line4 = ax2.plot(val_acc, ':', label="validation accuracy")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Cross Entropy Loss")

    ax2.set_ylabel("Accuracy")

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')


    magic_plt_show(filename)

def create_cm_lstm(model, dataloader, title, filename):
    """
    Creates the confution matrix for the given model
    """

    y_true_val, y_pred_val = eval_lstm_model_on_dataset(model, dataloader)

    cm = confusion_matrix(y_true_val, y_pred_val)

    plt.figure()
    plot_confusion_matrix(cm, [f"digit {i}" for i in range(10)], title=title)
    
    magic_plt_show(filename)

def accuracy_lstm_on_dataloader(model, dataloader):
    """
    Caclulates the accuracy of the model on the given dataloader
    """

    y_true_val, y_pred_val = eval_lstm_model_on_dataset(model, dataloader)

    return accuracy_score(y_true_val, y_pred_val)
