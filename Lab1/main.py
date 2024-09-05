from lib import *

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC


# Set a seed so that we always get the same results.
# This is not needed.
random.seed(12345)


####################
# STEP 1
####################

step_seperator(1)

train = np.loadtxt(Path('data/train.txt'))
test = np.loadtxt(Path('data/test.txt'))


X_train = train[:, 1:]
y_train = train[:, 0].astype("int64") # we cast the labels from floats to integers

X_test = test[:, 1:]
y_test = test[:, 0].astype("int64") # we cast the labels from floats to integers


####################
# STEP 2
####################


step_seperator(2)

show_sample(X_train, 131, "figures/step_2_digit.png")

####################
# STEP 3
####################

step_seperator(3)

plot_digits_samples(X_train, y_train, "figures/step_3_digits.png")

####################
# STEP 4
####################

step_seperator(4)

mean_0_10_10 = digit_mean_at_pixel(X_train, y_train, 0, (10,10))
print(f'Mean value of attributes of (10,10) pixel for digit 0 is {mean_0_10_10:.3f}')

####################
# STEP 5
####################

step_seperator(5)

variance_0_10_10 = digit_variance_at_pixel(X_train, y_train, 0, (10,10))
print(f'Variance value of attributes of (10,10) pixel for digit 0 is {variance_0_10_10:.3f}')

####################
# STEP 6
####################

step_seperator(6)

mean_0 = digit_mean(X_train, y_train, 0)
variance_0 = digit_variance(X_train, y_train, 0)

####################
# STEP 7
####################

step_seperator(7)

fig, ax = plt.subplots()
ax.imshow(mean_0.reshape((16,16)), cmap = 'gray')
ax.set_title('Digit 0 based on the\n mean values of the features')
ax.axis('off')
fig.tight_layout()

magic_plt_show("figures/step_7_mean_0.png")

####################
# STEP 8
####################

step_seperator(8)

# plot the mean value of the 0 digit

fig, ax = plt.subplots()
ax.imshow(variance_0.reshape((16,16)), cmap = 'gray')
ax.set_title('Digit 0 based on the\n variance values of the features')
ax.axis('off')
fig.tight_layout()

magic_plt_show("figures/step_8_variance_0.png")

# plot mean_0 and variance_0 side by side again

fig, ax = plt.subplots(1,2)
#fig.suptitle("comparison", fontsize=15)

ax[0].imshow(mean_0.reshape((16,16)), cmap = 'gray')
ax[0].set_title('Digit 0\n based on the mean values\n of the features')
ax[0].axis('off')

ax[1].imshow(variance_0.reshape((16,16)), cmap = 'gray')
ax[1].set_title('Digit 0\n based on the variance values\n of the features')
ax[1].axis('off')

fig.tight_layout()

magic_plt_show("figures/step_8_side_by_side.png")

####################
# STEP 9
####################

step_seperator(9)

####################
# Step 9.a compute the mean and variance for each digit
####################

# X_mean is a matrix of shape (10, 256)
# each row corresponds to a class
# and each column corresponds to a pixel
X_mean = np.zeros((10, 256))

# X_var is used exactly the same way as X_mean is used
X_var = np.zeros((10, 256))

# compute the mean and variance for each digit.
# The mean and variance for the digit 0 is recomputed but it's ok
for i in range(10):
    X_mean[i] =  digit_mean(X_train, y_train, i)
    X_var[i] = digit_variance(X_train, y_train, i)

####################
# Step 9.b plot the mean for each digit
####################


fig, ax = plt.subplots(2,5)
ax = ax.reshape(-1) # make ax a vector (it was a matrix)
for i in range(10):
  ax[i].imshow(X_mean[i].reshape((16,16)), cmap = 'gray')
  ax[i].set_title(f'{i}')
  ax[i].axis('off')

fig.suptitle("All the digits\nbased on the mean values of the features", fontsize=15)

fig.tight_layout()
magic_plt_show("figures/step_9_b_means.png")

#########
# STEP 10
#########

step_seperator(10)

#table of euclidean distances from each 0-9 digit's mean features to X_test[101]
euclidean_distances_digit_101 = [euclidean_distance(X_mean[i], X_test[101]) for i in range(10)]

# find the category
digit_101_category = np.argmin(euclidean_distances_digit_101)
print(f'According to euclidean distance, digit no 101 of test data is classified as the digit {digit_101_category}.')

#check known labels of test set to find out if the above classification is correct or not
eval_101 = "correct" if digit_101_category == y_test[101] else "wrong"
print("This classification is " + eval_101 + ". \nThe digit no 101 according to test set labels is the digit " + str(int(y_test[101])) +".")

# plot the sample at index 101

fig, ax = plt.subplots()
ax.imshow(X_test[101].reshape((16,16)), cmap = 'gray')
ax.set_title('Sample of the Test Set\nat index 101')
ax.axis('off')
fig.tight_layout()

magic_plt_show("figures/step_10_sample_101_test_set.png")

#############
# STEP 11
#############

step_seperator(11)

# Classify every sample of the test set
predictions_test_set = euclidean_distance_classifier(X_test, X_mean)

# calculate the accuracy of the classifier on the test set
correct_classifiactions = np.sum(predictions_test_set == y_test)
euclidean_accuracy = correct_classifiactions / len(y_test)

print(f'The accuracy of the Euclidean classifier is {euclidean_accuracy*100:.3f} %.')

#########
# STEP 12
#########

step_seperator(12)

# the code is in lib.py
# we do not have something to execute in this step

#########
# STEP 13
#########

step_seperator(13)

#a  ---------------------------

# train the Euclidean Classifier we implemented
euclidean_clf = EuclideanDistanceClassifier()
euclidean_clf.fit(X_train, y_train)

# calculate the accuracy
euclidean_score = evaluate_classifier(
    euclidean_clf,
    X_train, 
    y_train,
    folds=5
  )

print(f"Euclidean Classifier Cross Validation score (accuracy): {euclidean_score*100:.3f} %.")

#b  ---------------------------

# plot the decision boundaries
plot_clf(
      euclidean_clf,
      X_train,
      y_train,
      labels=[str(i) for i in range(10)],
      filename="figures/step_13_b_decision_plot.png"
  )


#c  ---------------------------

# calculate the learning curve for the euclidean classifer
train_sizes, train_scores, test_scores = learning_curve(
    euclidean_clf, X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(.01, 1.0, 12))

# plot the curve
plot_learning_curve(
        train_scores,
        test_scores,
        train_sizes,
        ylim=(.6, 1),
        filename="figures/step_13_c_learning_curves.png"
    )


#############
# STEP 14
#############

step_seperator(14)

# calculate the priors
priors = calculate_priors(X_train, y_train)

for _class, prior in enumerate(priors):
  print("P(digit_label={}) = {:.3f}".format(_class, prior))



#########
# Step 15
#########

step_seperator(15)

# a ~~~~~~

#create custom naive bayes classifier
custom_nb_clf = CustomNBClassifier()

#fit to train set
custom_nb_clf.fit(X_train, y_train)

# b ~~~~~~
#calculate accuracy score 
custom_nb_score = custom_nb_clf.score(X_test, y_test) 
print(f"Custom naive Bayes test accuracy = {custom_nb_score*100:.2f} %.")

# c ~~~~~~~~
#create naive bayes classifier (sklearn class)
#specify var_smoothing=1e-5 so results are comparable to our implementation of naive bayes classifier class
sklearn_nb_clf = GaussianNB(var_smoothing=1e-5)
sklearn_nb_clf.fit(X_train, y_train)

sklearn_nb_score = sklearn_nb_clf.score(X_test, y_test) 
print(f"sklearn naive Bayes test accuracy = {sklearn_nb_score*100:.2f} %.")


#########
# Step 16
#########

step_seperator(16)

custom_nb_clf_unit_var = CustomNBClassifier(use_unit_variance=True)
custom_nb_clf_unit_var.fit(X_train, y_train)

# b ~~~~~~

custom_nb_score = custom_nb_clf_unit_var.score(X_test, y_test) 
print(f"Custom naive Bayes with unit variance test accuracy = {custom_nb_score*100:.2f} %.")


#########
# Step 17
#########

step_seperator(17)
#print accuracy for SVM (rbf, linear, sigmoid, poly kernel), 1-NN, 5-NN, 15-NN, 51-NN & naive bayes classifiers 
# (in order of decreasing accuracy score)

print(" Model Name          | Accuracy")
print("------------------------------------------------------------------------")

evaluate_linear_svm_classifier(X_train, y_train)
evaluate_rbf_svm_classifier(X_train, y_train)
evaluate_poly_svm_classifier(X_train, y_train)
evaluate_sigmoid_svm_classifier(X_train, y_train)

evaluate_knn_classifier(X_train, y_train, 1)
evaluate_knn_classifier(X_train, y_train, 5)
evaluate_knn_classifier(X_train, y_train, 15)
evaluate_knn_classifier(X_train, y_train, 51)

evaluate_sklearn_nb_classifier(X_train, y_train)
evaluate_custom_nb_classifier(X_train, y_train, unit_var=False)
evaluate_custom_nb_classifier(X_train, y_train, unit_var=True)
print("------------------------------------------------------------------------")


#########
# Step 18
#########

step_seperator(18)

#dictionary of available classifiers (their names and the respective sklearn class)
models = {
    "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-5),
    "1-NN":  KNeighborsClassifier(n_neighbors=1),
    "5-NN":  KNeighborsClassifier(n_neighbors=5),
    "15-NN": KNeighborsClassifier(n_neighbors=15),
    "51-NN": KNeighborsClassifier(n_neighbors=51),
    "SVM linear kernel":  SVC(kernel="linear", probability=True),
    "SVM poly kernel":    SVC(kernel="poly", probability=True),
    "SVM rbf kernel":     SVC(kernel="rbf", probability=True),
    "SVM sigmoid kernel": SVC(kernel="sigmoid", probability=True)
    }


#calculate mispredictions per digit for every classifier calling mispredict

print('| Classifier           | Digits in order of decreasing misprediction rate |')
print("| :----------------:   | :-----------------------------------------------:|")

for name, m in models.items():
    mispred = mispredictions(m, X_train, y_train, X_test, y_test)
    print(f"| {name: <20} | {mispred}                            |")

print("| :----------------:   | : ----------------------------------------------:|")

# a ~~~~~~
##################################
##### VOTING CLASSIFIER ##########
##################################


#accuracy scores list for the various estimators the voting classifier is going  to combine voting c
#list of 2-element lists (first element accuracy with hard and second with soft voting method)
vot_exp = []


##experiments table 
estim_exp = []


#different classifier combinations

#0
estim_exp.append([("SVM poly kernel",SVC(kernel='poly', probability=True)), 
            ("1-NN", models["1-NN"]),
          #  ("SVM linear kernel", SVC(kernel='linear', probability=True))
])

#1
estim_exp.append([("SVM poly kernel",SVC(kernel='poly', probability=True)), 
            ("1-NN", models["1-NN"]),
            ("SVM linear kernel", SVC(kernel='linear', probability=True))
])

#2
estim_exp.append([("SVM poly kernel",SVC(kernel='poly', probability=True)), 
            ("5-NN", models["5-NN"]), 
            ("1-NN", models["1-NN"])
])

#3
estim_exp.append([("SVM poly kernel",SVC(kernel='poly', probability=True)), 
            ("SVM rbf kernel", SVC(kernel='rbf', probability=True))
            ,("1-NN", models["1-NN"])
])
# estim_exp.append([("SVM poly kernel",SVC(kernel='poly', probability=True)), 
#             ("SVM rbf kernel", SVC(kernel='rbf', probability=True))
#             ,("1-NN", models["1-NN"]), 
#             ("SVM linear kernel",SVC(kernel='linear', probability=True))
# ])
#list of lists of weights corresponding to each of the above set of estimators 
#estimator set #0 <--> weights[0], ...

weights = [ [8,4],
            #weights for SVM poly kernel, 1-NN respectively
            [8, 2, 1], 
            #SVM poly kernel, 1-NN, SVM linear kernel
            [8, 2, 1],
            #SVM poly kernel, 5-NN, 1-NN
            [8, 4, 2]
            #SVM poly kernel, SVM rbf kernel, 1-NN

            #,[10, 1, 1, 1]

]

#calculate accuracy score of voting classifier using above estimators
for i, estimator in enumerate(estim_exp):
  vot_exp.append(
    [evaluate_voting_classifier(estimator, X_train, y_train,'hard', weights[i]),
    evaluate_voting_classifier(estimator, X_train, y_train,'soft', weights[i])])


#print results
# escape codes for bold letters
start = "\033[1m"
end = "\033[0m"

for i, exp in enumerate(vot_exp):
  estimators = ""
  for classif in estim_exp[i]:
        estimators += classif[0]
        estimators += ", "
  print(f'Voting classifier using classifiers: {estimators}hard voting method has accuracy of {start}{exp[0]*100:.3f}{end} %.')
  print(f'Voting classifier using classifiers: {estimators}soft voting method has accuracy of {start}{exp[1]*100:.3f}{end} %.\n')

# b ~~~~~~
##################################
##### BAGGING CLASSIFIER #########
##################################


# ======== Bagging on SVM with Poly Kernel =========

print("Bagging: SVM with poly kernel:")

#specify svm poly kernel as base estimator
base_clf = SVC(kernel='poly', probability=True)

#number of estimators of bagging classifier to test
bagg_exp = [5, 10, 15, 25]
for n in bagg_exp:
  bagg_score = evaluate_bagging_classifier(base_clf, n, X_train, y_train)
  print(f'Bagging classifier with {n} base estimators has accuracy of {start}{bagg_score*100:.3f}{end} %')


# ======== Bagging on SVM with Linear Kernel ========

print("\n\n")
print("Bagging: SVM with linear kernel:")
#specify svm linear kernel as base estimator
base_clf = SVC(kernel='linear', probability=True)
for n in bagg_exp:
  bagg_score = evaluate_bagging_classifier(base_clf, n, X_train, y_train)
  print(f'Bagging classifier with {n} base estimators has accuracy of {start}{bagg_score*100:.3f}{end} %')



#########
# Step 19
#########

step_seperator(19)

# a ~~~~~~~~
# b ~~~~~~~~~


# configs is a list of tuples of the form (layers, epochs)
# layers: is a list describing the size of each layer
#         layers[0] should always be 256 because its the input size
#         layers[1] should always be 10 because that is the number of the classes
# epochs: is the number of epochs that the model is trained for

configs = [
    ([256, 10], 10),
    ([256, 100, 10], 10),
    ([256, 100, 100, 10], 20),
    ([256, 1000, 10], 20)
    ]

for layers, epochs in configs:


    pytorch_model = PytorchNNModel(layers, epochs)
    pytorch_model.fit(X_train, y_train)

    test_accuracy = pytorch_model.score(X_test, y_test)


    print("\n")
    print(f"Model config is layers={layers}, epochs={epochs}")
    print("test accuracy =", test_accuracy)
    print("\n")

