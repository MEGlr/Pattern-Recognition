from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, KFold, learning_curve, train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import random
import statistics as stats

import torch
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path


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


def show_sample(X, index, filename=""):
    """Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    """

    plt.title(str(index) + "-th digit")
    plt.axis('off')
    plt.imshow(np.reshape(X[index], (16, 16)), cmap="gray")

    magic_plt_show(filename);


def plot_digits_samples(X, y, filename=""):
    """Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    """

    # use digit 0-9 to organize y's data represented by index in table indices_for_digit
    # indices_for_digit[i] = list of indices of the samples which belong to class i
    indices_for_digit = [[] for i in range(10)]
    for i, digit in enumerate(y):
        indices_for_digit[int(digit)].append(i)

    # select one index from each digit in random (using random.choise function)
    # digit_samples[i] = the index of the digit to be ploted which belongs to class i
    digit_samples = []
    for i in range(10):
        digit_samples.append(random.choice(indices_for_digit[i]))

    # plot the randomly chosen samples
    fig, ax = plt.subplots(2, 5)
    ax = ax.reshape(-1) # place all axises on a vector
    
    fig.suptitle("Random Digits 0-9", fontsize=15)

    # plot each sample
    for i in range(10):
        ax[i].imshow(np.reshape(X[digit_samples[i]], (16, 16)), cmap="gray")
        ax[i].set_title("digit : {}".format(i))
        ax[i].set_axis_off()

    fig.tight_layout()
    magic_plt_show(filename);


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    """Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    """

    # digit_indices = list of indices of samples that belong to the class `digit`
    # `digit` is the parameter of the function
    digit_indices = []
    for i, digit_ in enumerate(y):
        if digit_ == digit:
            digit_indices.append(i)

    # pixel_values = It is a list of all the values the pixel of interest takes
    # pixel_values[i] = value for the pixel at position `pixel` at sample X[ digit_indices[i] ]
    pixel_values = []
    for index in digit_indices:
        pixel_values.append(X[index][pixel[0] * 16 + pixel[1]])

    # calculate the mean
    return stats.mean(pixel_values)


def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    """Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    """

    # digit_indices = list of indices of samples that belong to the class `digit`
    digit_indices = np.where(y == digit)[0]

    # pixel_values = all the values that the pixel of interest takes.
    # the same as in digit_mean_at_pixel()
    pixel_values = []
    for index in digit_indices:
        pixel_values.append(X[index][pixel[0] * 16 + pixel[1]])

    # calculate the variance
    return stats.variance(pixel_values)


def digit_mean(X, y, digit):
    """Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    """



    digit_indices = np.where(y == digit)[0]
    
    if (len(digit_indices) == 0):
        # if there are no samples in X for the class `digit`
        # return a 16x16 matrix with elements of value INF

        # We choose to return an "INF" matrix and not a
        # zero one so that the euclidean classifier
        # never classifies a sample to a category that is
        # not present in the train set.

        INF = 1e10
        means = np.full((16,16), INF)
    else:
        # if there are samples in X for the class `digit`
        # then just calculate the mean on each pixel.

        means = [
            [digit_mean_at_pixel(X, y, digit, (i, j)) for j in range(16)] for i in range(16)
        ]
        
    # return the mean values as a vector
    return np.array(means).reshape(-1)


def digit_variance(X, y, digit):
    """Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    """


    # calculate the variance on each pixel.
    variances = [
        [digit_variance_at_pixel(X, y, digit, (i, j)) for j in range(16)]
        for i in range(16)
    ]

    # return the variance values as a vector
    return np.array(variances).reshape(-1)


def euclidean_distance(s, m):
    """Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    """

    return np.sqrt(np.sum((s - m) ** 2))


def euclidean_distance_classifier(X, X_mean):
    """Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    """

    # prediction for each sample
    # predictions[i] = prediction for sample X[i]
    predictions = []

    for sample in X:
        # calculate the distances of the current sample to all the means
        euclidean_distances_digit_current = [euclidean_distance(sample, X_mean[i]) for i in range(10)]

        # classify using the argmin function
        pred = np.argmin(euclidean_distances_digit_current)

        # save the prediction
        predictions.append(pred)

    # just return the result
    return np.array(predictions)


class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """

        self.X_mean_ = np.zeros((10, 256))

        # compute the mean for each digit
        for i in range(10):
            self.X_mean_[i] =  digit_mean(X, y, i)

        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        
        # just a wrapper for the euclidean_distance_classifier function
        return euclidean_distance_classifier(X, self.X_mean_)

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """

        predictions = self.predict(X)
        euclidean_accuracy = (np.sum(predictions == y))/len(y)

        return euclidean_accuracy


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    clf.fit(X,y)
    # use the cross validation technique to calcualte the accuracy
    # of the classifier
    scores = cross_val_score(clf, X, y, 
                         cv=KFold(n_splits=folds, shuffle=True, random_state=42), 
                         scoring="accuracy", n_jobs=4)

    # return the mean of all the score that got calculated
    return np.mean(scores)

def plot_clf(clf, X, y, labels, filename=""):


    # We PCA to reduce to only 2 dimensions (initially we have 256 features)
    pca = PCA(n_components = 2)
    pca.fit(X)
    transformed_X = pca.transform(X)

    fig, ax = plt.subplots()

    # title for the plots
    title = ('Decision surface of Classifier')

    # Set-up grid for plotting.
    X0, X1 = transformed_X[:, 0], transformed_X[:, 1]
    
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                         np.linspace(y_min, y_max, 30))
    
    # we map the 2D space of xx, yy back to the original 256-D space
    # by calling pca.inverse_transform
    Z = clf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    color_set = list(mcolors.TABLEAU_COLORS.keys())

    out = ax.contourf(xx, yy, Z, colors=color_set, alpha=0.8, levels=[i-0.5 for i in range(10+1)])
    fig.colorbar(out, ticks=[i for i in range(0,10)])
    
    # plot the transformed samples onto the plain
    for i in range(len(labels)):
        ax.scatter(
            X0[y == i], X1[y == i],
            c=color_set[i],
            label=labels[i],
            s=60, alpha=0.9, edgecolors='k'
        )


    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
   
    magic_plt_show(filename)


def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1), filename=""):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    magic_plt_show(filename)

    
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """

    # calcluate how many samples are in each class
    # counts[i] = number of samples for class i
    counts = np.bincount(y)

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

        self.X_mean_ = np.zeros((10, 256))

        # compute the mean for each digit
        for i in range(10):
            self.X_mean_[i] =  digit_mean(X, y, i)


        ####################################################
        # Set the variance for each pixel and for each class
        ####################################################

        # by default use unit variance
        self.X_variance_ = np.ones((10,256))

        if not self.use_unit_variance:

            # if self.use_unit_variance == False
            # calculate the variabce

            for i in range(10):
                self.X_variance_[i] = digit_variance(X, y, i)

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
            direct = [self.direct_prob(sample, y) for y in range(10)]
            direct = np.array(direct)

            # the prediction for this sample
            pred = np.argmax(direct * self.priors)

            predictions.append(pred)


        return predictions

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """

        predictions = self.predict(X)
        euclidean_accuracy = (np.sum(predictions == y))/len(y)

        return euclidean_accuracy

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
                loc=self.X_mean_[y],
                scale=np.sqrt(self.X_variance_[y])
            )

        # and multiply all of them together
        return np.prod(p_pixel)



class FullyConnectedNeuralNetwork(torch.nn.Module):

    def __init__(self, LAYER_SIZE):
        super(FullyConnectedNeuralNetwork, self).__init__()

        # LAYER_SIZE[0] = input size, always 256
        # LAYER_SIZE[1] = the output size of the first layer
        # LAYER_SIZE[2] = the output size of the second layer
        # ...
        # LAYER_SIZE[-1] = the output size of the whole NN, always 10

        # tmp_layers is an array that will hold all the layers of the NN
        tmp_layers = []

        for layer_num in range(1, len(LAYER_SIZE)):
            tmp_layers.append(
                    torch.nn.Linear(LAYER_SIZE[layer_num-1], LAYER_SIZE[layer_num])
                )

            if(layer_num != len(LAYER_SIZE) - 1):
                # if we did not add the last linear layer add a relu layer
                tmp_layers.append( torch.nn.ReLU() )

        self.layers = torch.nn.ModuleList(tmp_layers)



    def forward(self, x):

        for layer in self.layers:
            # we cast to float because it otherwise throws
            # an error. (x is of type double without casting)
            x = layer(x.float())

        logits = x

        return logits

def eval_on_dataset(model, dataloader):
    """
    Evaluates the model on the given dataloader
    and returns the predictions and the true values.

    It returns a tuple (y_true, y_pred)

    where
        y_true: contains the true labels which are
            extracted from the dataloader
        y_pred: are the predictions of the model
            on the given dataset
    """

    y_pred = []
    y_true = []

    with torch.no_grad():

        for inputs, labels in dataloader:

            logits = model(inputs)

            predictions = torch.argmax(logits, dim=1)

            # append the predictions and the true labels
            y_pred += predictions.int().tolist()
            y_true += labels.int().tolist()

    return y_true, y_pred




class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, layers, epochs, batch_size=16, learning_rate=1e-3):
        # WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.

        # Some configuration variables
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size


        LEARNING_RATE = learning_rate
        
        self.model = FullyConnectedNeuralNetwork(layers)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr = LEARNING_RATE
            )

    def fit(self, X, y):
        # split X, y in train and validation set and wrap in pytorch dataloaders

        # make a 80% - 20% split of the data
        X_train, X_val, Y_train, Y_val = train_test_split(
                X, y,
                test_size=0.8, shuffle=True, random_state=42
            )

        # Create PyTorch dataset
        train_dataset = TensorDataset(
                torch.tensor(X_train),
                torch.tensor(Y_train)
            )

        val_dataset = TensorDataset(
                torch.tensor(X_val),
                torch.tensor(Y_val)
            )

        # Create the initializers
        train_loader = DataLoader(
                train_dataset,
                batch_size = self.BATCH_SIZE,
                shuffle = True
            )

        val_loader = DataLoader(
                val_dataset,
                batch_size = self.BATCH_SIZE,
                shuffle = True
            )

        # Train the model

        for epoch in range(self.EPOCHS):

            # set model to train mode, not needed in our case but ok
            self.model.train()

            # train on the whole train set
            for inputs, labels in train_loader:

                # reset the accumulated gradient
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.model(inputs)

                # calculate loss
                loss = self.criterion(outputs, labels)

                # compute the gradients
                loss.backward()

                # update the model parameters
                self.optimizer.step()

            # set model to evaluation mode, not needed
            self.model.eval()

            # calculate the accuracy on the train set
            y_true, y_pred = eval_on_dataset(self.model, train_loader)
            train_acc = accuracy_score(y_true, y_pred)

            # calculate the accuracy on the validation step
            y_true, y_pred = eval_on_dataset(self.model, val_loader)
            val_acc = accuracy_score(y_true, y_pred)

            print(f"Epoch {epoch: <3}: train accuracy = {train_acc:.3f}, validation accuracy = {val_acc:.3f}")

        return self

    def predict(self, X):
        # TODO: wrap X in a test loader and evaluate

        test_dataset = TensorDataset(
                torch.tensor(X)
            )

        test_loader = DataLoader(
                test_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=False
            )

        # the predictions to return
        y_pred = []

        # no needed but ok
        self.model.eval()

        with torch.no_grad():

            for minibatch in test_loader:

                # minibatch is just a python array
                # with one element
                samples = minibatch[0]

                logits = self.model(samples)

                pred = torch.argmax(logits, dim=1)

                # append the predictions to the list
                y_pred += pred.int().tolist()

        return np.array(y_pred)

    def score(self, X, y):
        # Return accuracy score.

        predictions = self.predict(X)
        
        # we just use the sklean implementation for the accuracy
        return accuracy_score(predictions, y)



def evaluate_linear_svm_classifier(X, y, folds=5):
    """Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier

    no return value, it prints the accuracy score
    """
    #Create an svm with linear kernel
    clf = SVC(kernel="linear", probability=True)

    #calculate score (cross-validation)
    score = evaluate_classifier(
        clf,
        X, 
        y,
        folds=5
    )
    
    #print accuracy score of clf
    name = "SVM linear"
    print(f"{name: <20} | {score*100:.3f} %")



def evaluate_rbf_svm_classifier(X, y, folds=5):
    """Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier

    no return value, it prints the accuracy score
    """
    #Create an svm with rbf kernel
    clf = SVC(kernel="rbf", probability = True)

    #calculate score (cross-validation)
    score = evaluate_classifier(
        clf,
        X, 
        y,
        folds=5
    )

    #print accuracy score of clf
    name = "SVM rbf"
    print(f"{name: <20} | {score*100:.3f} %")

def evaluate_poly_svm_classifier(X, y, folds=5):
    """Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier

    no return value, it prints the accuracy score
    """
    #Create an svm with poly kernel
    clf = SVC(kernel="poly", probability = True)

    #calculate score (cross-validation)
    score = evaluate_classifier(
        clf,
        X, 
        y,
        folds=5
    )

    #print accuracy score of clf
    name = "SVM poly"
    print(f"{name: <20} | {score*100:.3f} %")

def evaluate_sigmoid_svm_classifier(X, y, folds=5):
    """C and evaluate it using cross-validation
    Calls evaluate_classifier

    no return value, it prints the accuracy score
    """

    #Create an svm with sigmoid kernel
    clf = SVC(kernel="sigmoid", probability = True)

    #calculate score (cross-validation)
    score = evaluate_classifier(
        clf,
        X, 
        y,
        folds=5
    )

    #print accuracy score of clf
    name = "SVM sigmoid"
    print(f"{name: <20} | {score*100:.3f} %")


def evaluate_knn_classifier(X, y, K, folds=5):
    """Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier

    no return value, it prints the accuracy score
    """
    #Create a knn (k is given as a parameter)
    clf = KNeighborsClassifier(n_neighbors=K)

    #calculate score (cross-validation)
    score = evaluate_classifier(
        clf,
        X, 
        y,
        folds=5
    )
    
    #print accuracy score of clf
    name = f"{K}-NN"
    print(f"{name: <20} | {score*100:.3f} %")



def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier

    no return value, it prints the accuracy score
    """
    #create sklearn naive bayes classifier
    #we specify variance smoothing parameter (value that gets added to variance value of every pixel)
    #to be the same to our implementation of naive bayes classifier

    clf = GaussianNB(var_smoothing=1e-5)

    #calculate score (cross-validation)
    score = evaluate_classifier(
        clf,
        X, 
        y,
        folds=5
    )

    #print accuracy score of clf
    name = "sklearn Naive Bayes"
    print(f"{name: <20} | {score*100:.3f} %")



def evaluate_custom_nb_classifier(X, y, unit_var, folds=5):
    """Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier

    no return value, it prints the accuracy score
    """
    #Create a custom naive bayes classifier (variance of all pixels is set = 1)
    clf = CustomNBClassifier(use_unit_variance=unit_var)

    #calculate score (cross-validation)
    score = evaluate_classifier(
        clf,
        X,
        y,
        folds=5
    )

    #print accuracy score of clf
    name = "Custom NB" + ("" if unit_var == False else " unit var") 
    print(f"{name: <20} | {score*100:.3f} %")
    


# def evaluate_euclidean_classifier(X, y, folds=5):
#     """Create a euclidean classifier and evaluate it using cross-validation
#     Calls evaluate_classifier
#     """
#     # we already have evaluated the euclidean classifier
#     raise NotImplementedError


# def evaluate_nn_classifier(X, y, folds=5):
#     """Create a pytorch nn classifier and evaluate it using cross-validation
#     Calls evaluate_classifier
#     """
#     raise NotImplementedError


def evaluate_voting_classifier(classifiers, X, y, voting_method, w, folds=5):
    """Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """

    #create clf 
    clf = VotingClassifier(estimators = classifiers, voting = voting_method, n_jobs=4)
    
    #array of scores calculated by evaluate for each estimator 
    scores = evaluate_classifier(clf, X, y, folds)
    
    #return mean value of all individual scores 
    return np.mean(scores)


def evaluate_bagging_classifier(base_clf, N, X, y, folds=5):
    """Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    # The bags will contain 80 % * len(X) samples
    #create clf
    clf = BaggingClassifier(base_clf, N, max_samples=0.8, n_jobs=4)

    #array of scores calculated by evaluate for each estimator 
    scores = evaluate_classifier(clf, X, y, folds) 

    #return mean value of all individual scores 
    return np.mean(scores)

def mispredictions(clf, X_train, y_train, X_test, y_test):
    """Returns digits in order of decreasing misprediction count for classifier clf on X and y
       a digit d is considered to be mispredicted if a sample labeled d but is predicted as a different digit in test set

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.array): The digits in order of decreasing misprediction count (10 elements = #digits)
    """

    #fit classifier to the train set
    clf.fit(X_train, y_train)

    #predict on test set
    pred = clf.predict(X_test)
    
    #initialize misprediction count for each digit 
    mispred = [0]*10

    #loop though test set
    N = len(y_test)
    for i in range(N):
        #for each prediction check is it's wrong or not
        if y_test[i] != pred[i]: 
            # misprediction : did not recognize digit y_test[i] for which one it is
            
            #increase misprediction count for digit y_test[i]
            mispred[int(y_test[i])] += 1   
    mispred = np.array(mispred)
    return np.argsort(mispred)[::-1] # return sorted array of digits according to decreasing mispredictions count
