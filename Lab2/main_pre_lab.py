import sys

import multiprocessing as mp

from lib import *

from helper_scripts.parser import *
from helper_scripts.plot_confusion_matrix import *

import pickle

N1 = 6
N2 = 2

RUN_IN_PARALLEL=True

# =============== STEP 2 ===============

step_seperator(2)

wavs, speaker_ids, labels, Fs = data_parser()

# =============== STEP 3 ===============

step_seperator(3)

# The indexing is as follows
# features[number of the utterance][number of the frame][mumber of the mfcc]
features = extract_features(wavs, 13, Fs=Fs)

# =============== STEP 4 ===============

step_seperator(4)


# create the histograms
for d in [N1, N2]:

    plot_hist_step_4(
            features,
            labels,
            for_digit = d,
            plot_title = f"Histogram for the digit {d}",
            plot_filename = f"figures/step_4_hist_digit{d}.png"
        )


#part b -------------------

#extract Mel Filterbank Spectral Coefficients (MFSCs) for all wavs

features_mfscs = extract_mfsc(wavs, 13, Fs=Fs)

#MFCCs για την κάθε εκφώνηση
plot_cor(   
            features,
            labels,
            for_digits = [N1, N2],
            plot_title = "MFCC",
            plot_filename=f"figures/step_4_cor_mfcc.png"
)

#MFSCs για την κάθε εκφώνηση
plot_cor(   
            features_mfscs,
            labels,
            for_digits = [N1, N2],
            plot_title = "MFSC",
            plot_filename=f"figures/step_4_cor_mfsc.png"
)

# =============== STEP 5 ===============

step_seperator(5)

feat_vector = np.zeros((len(wavs), 6*13))

feat_vector = []
#13 x 2 elements for the means and stds of MFCCs for each feature
#other 13 x 2 for deltas and 13 x2 for delta-deltas

for i in range(len(wavs)):
    #features 3rd dimention is of size 13 x 3, where the 1rst 13 
    #are the MFCCs, the 2nd 13 are the deltas and the rest 13 are the delta-deltas

    # features[i][:, :13] : mfcc's for t-th utterance
    #(73 x 13) array (75 = # frames) 
    #we want to calculate the average for all the frames
    #for each of the 13 features
    #the feature vector in going to have 
    #the means of mfccs, delta & delta-deltas first followed 
    #by their stds

    #TODO: delete the following comments (do not 
    # care about the ordering of features)

    # mfcc_mean = np.mean(features[i][:, :13], axis = 0)
    # mfcc_std = np.std(features[i][:, :13], axis = 0)
    # delta_mean = np.mean(features[i][:, 13:26], axis = 0)
    # delta_std = np.std(features[i][:, 13:26], axis = 0)
    # deltadelta_mean = np.mean(features[i][:, 26:], axis = 0)
    # deltadelta_std = np.std(features[i][:, 26:], axis = 0)
    # conc_ = np.concatenate((mfcc_mean, mfcc_std, delta_mean, delta_std, deltadelta_mean, deltadelta_std))
    # feat_vector.append(conc_)

    feat_vector.append( 
        np.concatenate(
            (np.mean(features[i], axis = 0), 
             np.std(features[i], axis = 0)
            )
        )
    )

feat_vector = np.array(feat_vector)
#print(feat_vector.shape)
scatter(feat_vector, labels, "2 first dimentions of feature vector", "figures/step_5.png")


# =============== STEP 6 ===============

step_seperator(6)
# 2D ----------------------------------------------------
#calculate the PCA (n_components = 2 for 2D scatter plot)
pca_2 = PCA(n_components=2)
X_2D = pca_2.fit_transform(feat_vector)
scatter(X_2D, labels, "feature vector (after PCA) - 2D", "figures/step_6_2D_scatter_plot.png")

# 3D ----------------------------------------------------
#calculate the PCA (n_components = 3 for 3D scatter plot)
pca_3 = PCA(n_components=3)
X_3D = pca_3.fit_transform(feat_vector)
scatter(X_3D, labels, "feature vector (after PCA ) - 3D", "figures/step_6_3D_scatter_plot.png", dim = 3)

print('Variance Ratio (preserved) after PCA (# components = 2):', np.round(sum(pca_2.explained_variance_ratio_)*100, 2), "%")
print('Variance Ratio (preserved) after PCA (# components = 3):', np.round(sum(pca_3.explained_variance_ratio_)*100, 2), "%")
print('Variance Ratio (preserved) by each of the first 3 PC :',pca_3.explained_variance_ratio_)


# =============== STEP 7 ===============

step_seperator(7)


#dictionary with classifier we want to try
classifiers = {
    "NB": GaussianNB(var_smoothing=10e-5) ,  #we choose var_smoothing to be equal to the value defined in our implementation
    "CustomNB" : CustomNBClassifier(),
    "SVM (poly)": SVC(kernel='poly'),
    "SVM (linear)": SVC(kernel='linear'),
    "SVM (rbf)": SVC(kernel='rbf'),
    "1-NN": KNeighborsClassifier(n_neighbors=1),
    "5-NN": KNeighborsClassifier(n_neighbors=5),
    "15-NN": KNeighborsClassifier(n_neighbors=15),
    "50-NN": KNeighborsClassifier(n_neighbors=50),
    "LR" : LogisticRegression(max_iter=100000),

}


#split dataset in train and test
#we define random_state so the split does not change with every run.
rand = 7
X_train, X_test, y_train, y_test = train_test_split(feat_vector,labels, test_size=0.3, random_state=rand)

print("a priori :",  calculate_priors(X_train, y_train)*100)
print("--------------------------")

#list of accuracies for each classifier


accuracies = {}

# Normalize Data
transformer = StandardScaler().fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)


#train model, predict and print accuracy on test set
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies[clf_name] = acc

accuracies = dict(sorted(accuracies.items(), key=lambda x:x[1], reverse=True))

for clf_name, clf in accuracies.items():
    acc = accuracies[clf_name]
    print(f"Accuracy Score for {clf_name}: {np.round(acc*100, 2)} %")



# BONUS -------------------------------------------------

#EXTRA FEATURES

# vector als including extra features we are going to test 
feat_vector_extra = []
print("----------------------------------------------------------")
print("\nAccuracies after adding zero crossings to feature vector:")
# 1) calculate zero crossings
window = 25 * Fs // 1000 # 25 msec
step = 10 * Fs // 1000 

for i in range(len(wavs)):
    #zero crossing rate for each frame 
    z_cross = librosa.feature.zero_crossing_rate(
        wavs[i], 
        frame_length = window, 
        hop_length = window - step)

    feat_vector_extra.append(np.concatenate((feat_vector[i], [np.mean(z_cross)]))) 

feat_vector_extra = np.array(feat_vector_extra)
#print(feat_vector_extra.shape)

#TODO : add extra features to test

#79 #features used for classification
#split dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(feat_vector_extra,labels, test_size=0.3, random_state=rand)



accuracies = {}

# Normalize Data
transformer = StandardScaler().fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

#train model, predict and print accuracy on test set
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies[clf_name] = acc

accuracies = dict(sorted(accuracies.items(), key=lambda x:x[1], reverse=True))

for clf_name, clf in accuracies.items():
    acc = accuracies[clf_name]
    print(f"Accuracy Score for {clf_name}: {np.round(acc*100, 2)} %")



# =============== STEP 8 ===============


step_seperator(8)


# generate sequence pairs
x, y = gen_data_step_8(N=200)


# plot 4 of them
num_to_plot = 4

plt.figure(figsize=(10, 10))
for i in range(0, num_to_plot):
    plt.subplot(num_to_plot, 1, i+1)
    plt.title(f"Sequence pair #{i+1}")
    plt.plot(x[i], label="sin")
    plt.plot(y[i], label="cos")
    plt.legend()

magic_plt_show("figures/step_8_samples.png")



# history: the the MSE score on the test set
# and tehe validation set. It will be filled
# up later and then used to create some
# learning curves.
history = dict()

# examples: will hold the prediction of the
# model on a single sample. It will be 
# ploted later.
examples = dict()


# train all the models
for rec_layer in ["RNN", "LSTM", "GRU"]:

    print(f"\n\nNow training using a {rec_layer}\n")

    #  create the model
    model = RecursiveCosPredictor(
            rec_layer, 
            hidden_dim=5,
            seq_len=10
        )

    # use MSE as the train loss
    criterion = nn.MSELoss()

    # the optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train the model
    history[rec_layer], examples[rec_layer] = train_regression_model(
            x, y,
            0.8, # train split 80% - 20%
            model, 
            criterion,
            optimizer,
            val_criterion=mean_squared_error,
            batch_size=10,
            epochs=120)

# plot the learning curves
print("\nplotting learning curves:")

for rec_layer in ["RNN", "LSTM", "GRU"]:

    train_loss, val_loss = history[rec_layer]

    plot_leaning_curves(
            train_loss, val_loss,
            f"Loss curves (MSE) using {rec_layer}",
            f"figures/step_8_lc_{rec_layer}.png"
        )

# Make subplots for each case.
# Each subplot will plot the input and the predition
# of the model.

print("\nplotting samples:")
plt.figure(figsize=(10, 10))

for i, rec_layer in enumerate(["RNN", "LSTM", "GRU"]):
    for j in range(3):

        x            = examples[rec_layer][0][j]
        ground_truth = examples[rec_layer][1][j]
        prediction   = examples[rec_layer][2][j]


        plt.subplot(3, 3, 3*i + j + 1)

        subplot_example(
                x, ground_truth, prediction,
                f"{rec_layer}",
            )

    magic_plt_show(f"figures/step_8_sample_predictions.png")

