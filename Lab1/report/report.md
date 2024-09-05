---
title: Αναφορά Πρώτης Εργαστηριακής Άσκησης
author:
- Μαρία Ηώς Γλάρου el18176
- Παναγιώτης Μενεγάκης el18052
mainfont: 'Times New Roman'
monofont: 'Courier New'
geometry: margin=1in
documentclass: report
header-includes: |
	\usepackage{cancel}
	\usepackage{float}
	\floatplacement{figure}{H}
	\floatplacement{table}{H}
	\usepackage{fvextra}
	\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}	

toc: true
urlcolor: blue
---

\newpage

## Σχετικά με τον κώδικα

Εν γένη έχουμε χρησιμοποιήσει το `lib.py` αρχείο που μας δόθηκε και
συμπληρώσαμε σε αυτό τις υλοποιήσεις. Το `main.py` είναι το script που
πρέπει να εκτελεστεί για να γίνουν όλοι οι υπολογισμοί για κάθε βήμα. Το
μόνο που χρειάζεται έτσι ώστε το script να εκτελεστεί επιτυχώς είναι η
ύπαρξη του φακέλου `data` που περιέχει τα data sets. Ο φάκελος `data` πρέπει
να βρίσκεται στον ίδιο φάκελο με το αρχείο `main.py`.

## Βήμα 1

Χρησιμοποιήσαμε την συνάρτηση `np.loadtxt()` της βιβλιοθήκης numpy για να διαβάσουμε τα test και train δεδομένα από τα αρχεία txt που δίνονται όπως περιγράφεται στην εκφώνηση. Επίσης για να είναι τα labels τύπου integer τα κάναμε cast με την χρήση της `astype()`.

## Βήμα 2

Το ψηφίο στην θέση 131 του dataset φαίνεται παρακάτω:

![Το sample στην θέση 131 του train set](../figures/step_2_digit.png){width=40%}

## Βήμα 3

Παρακάτω φαίνεται ένα τυχαία επιλεγμένο δείγμα για κάθε κατηγορία από το `X_train`:

![Τυχαία δείγματα για κάθε κατηγορία από το train set](../figures/step_3_digits.png){width=70%}

## Βήμα 4

Η μέση τιμή για το pixel (10,10)
για το ψηφίο 0 είναι $-0.504$.

Αυτό φαίνεται από την έξοδο του προγράμματος:

```{.html}
Mean value of attributes of (10,10) pixel for digit 0 is -0.504
```

## Βήμα 5

Ομοίως υπολογίζουμε την διασπορά για το pixel (10,10) για το ψηφίο 0 με βάση τα train δεδομένα οπότε προκύπτει ότι ισούται με $0.525$. 

Αυτό φαίνεται από την έξοδο του προγράμματος:

```{,html}
Variance value of attributes of (10,10) pixel for digit 0 is 0.525
```

## Βήμα 6

Υπολογίζουμε την μέση τιμή και την διασπορά για κάθε pixel του ψηφίου 0. Οι τιμές θα μας χρησιμεύσουν στα βήματα 7, 8.
Οι υπολογισμοί αυτοί έγιναν με την χρήση του train set.

## Βήμα 7

Το "μέσο" μηδέν όπως υπολογίστηκε από το βήμα 6 φαίνεται παρακάτω:

![To "μέσο" ψηφίο 0](../figures/step_7_mean_0.png){width=40%}

Όπως βλέπουμε το μέσο μηδενικό
είναι απλά μια θολή μορφή του
ψηφίου μηδέν. Αυτό συμβαίνει γιατί
έχουμε υπολογίσει την μέση τιμή όλων των ψηφίων μηδέν στο train set.

## Βήμα 8

Το μηδέν απεικονίζεται με βάση την τιμή της διασποράς σε κάθε pixel που υπολογίστηκε στο βήμα 6 και το αποτέλεσμα φαίνεται παρακάτω.

![To ψηφίο 0 με βάση τις τιμές των διασπορών](../figures/step_8_variance_0.png){width=40%}

### Σύγκριση μέσης τιμής και διασποράς των pixel για το ψηφίο 0

Παρακάτω φαίνεται η μέση τιμή του ψηφίου μηδέν αλλά και η διασπορά του σε ένα κοινό figure.

![To ψηφίο 0 με βάση τις τιμές των διασπορών](../figures/step_8_side_by_side.png){width=70%}


Τα μηδενικά δείγματα του train set συμφωνούν στις τιμές των περισσότερων pixels. Μεγαλύτερες διαφοροποιήσεις παρουσιάζονται στα pixels που ορίζουν την "περιφέρεια" του ψηφίου 0 (λόγω ελαφρών περιστροφών, του φάρδους του, του πλάτους της γραμμής κτλ). 

Για αυτό η μέση τιμή των pixel έχει μεγάλες τιμές στο "εσωτερικό" του ψηφίου 0 και η μέση τιμή φθίνει καθώς κινούμαστε προς την περιφέρεια. Αντιθέτως η διασπορά των τιμών των pixel είναι μικρή στο εσωτερικό του ψηφίου 0 αλλά λαμβάνει μεγαλύτερες τιμές στην περιφέρεια του (όπου εμφανίζονται και οι διαφοροποιήσεις στο train set).

## Βήμα 9

### Ερώτημα (α)

Σε αυτό το βήμα υπολογίσαμε την μέση τιμή και την διασπορά των pixels (των χαρακτηριστικών στην περίπτωση μας) με την χρήση των συναρτήσεων `digit_mean` και `digit_variance`. Ο υπολογισμός αυτός έγινε με την χρήση του train set.

### Ερώτημα (β)

Παρακάτω φαίνονται όλα τα ψηφία σχεδιασμένα βάση των μέσων τιμών τους:

![Η "μέση" εικόνα κάθε ψηφίου](../figures/step_9_b_means.png){width=70%}

## Βήμα 10

Σε αυτό το βήμα υπολογίζουμε την ευκλείδεια απόσταση του sample στην θέση 101 με τις μέσες τιμές όλων των ψηφίων με την χρήση της συνάρτησης `euclidean_distance` της `sklearn`. Ύστερα ταξινομούμε το δείγμα στην πιο κοντινή μέση τιμή με την χρήση της συνάρτησης `argmin`. Ο κώδικας φαίνεται παρακάτω:

```{.python}
euclidean_distances_digit_101 = [euclidean_distance(X_mean[i], X_test[101]) for i in range(10)]

digit_101_category = np.argmin(euclidean_distances_digit_101)
print(f'According to euclidean distance, digit no 101 of test data is classified as the digit {digit_101_category}.')

#check known labels of test set to find out if the above classification is correct or not
eval_101 = "correct" if digit_101_category == y_test[101] else "wrong"
print("This classification is " + eval_101 + ". \nThe digit no 101 according to test set labels is the digit " + str(int(y_test[101])) +".")

```

Τελικά το sample στην θέση 101 του test set ταξινομείται ως ψηφίο 0. To sample στην θέση 101 φαίνεται παρακάτω:

![Το sample στην θέση 101 του test set](../figures/step_10_sample_101_test_set.png){width=40%}


Σύμφωνα με το test set αυτό είναι το ψηφίο 6, άρα το ψηφίο ταξινομήθηκε λανθασμένα. Βέβαια είναι κάπως λογικό το sample αυτό να κατηγοριοποιηθεί λανθασμένα. Αν το παρατηρήσουμε το "εξωτερικό" κομμάτι του ψηφίου μοιάζει πολύ με μηδέν, συνεπώς θα έχει αρκετά μικρή ευκλείδεια απόσταση από το μέσο ψηφίο 0.

Επίσης μπορούμε να καταλάβουμε ότι το sample στην θέση 101 κατηγοριοποιήθηκε λανθασμένα γιατί στην έξοδο τυπώνεται το εξής:

```{.html}
This classification is wrong.
```

## Βήμα 11

<!-- TODO: say something for the code -->
Για αυτό το βήμα ενθυλακώσαμε ουσιαστικά τον κώδικα που περιγράψαμε και φαίνεται στο βήμα 10 στην συνάρτηση `euclidean_distance_classifier`. Ο κώδικας
της συνάρτησης αυτής φαίνεται παρακάτω:

```{.python}
def euclidean_distance_classifier(X, X_mean):
    """Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    """
    predictions = []
    for sample in X:
       euclidean_distances_digit_current = [euclidean_distance(sample, X_mean[i]) for i in range(10)]
       pred = np.argmin(euclidean_distances_digit_current)
       predictions.append(pred)

    return np.array(predictions)
```

Επαναλαμβάνουμε συνεπώς την διαδικασία ταξινόμησης με βάση την ευκλείδεια απόσταση
για όλα τα ψηφία του test set. 

Ως ποσοστό επιτυχίας υπολογίζουμε την ακρίβεια (accuracy) δηλαδή τον λόγο του πλήθους των σωστών ταξινομήσεων προς το συνολικό πλήθος των δειγμάτων του test set. 

Η ταξινόμηση που κάνουμε οδηγεί σε ποσοστό επιτυχίας (accuracy) $81.42\%$ το
οποίο είναι ικανοποιητικό δεδομένης της απλότητας του μοντέλου. Για κάποιες
από τις λανθασμένες ταξινομήσεις εκτιμούμε ότι μπορεί να οφείλεται η θεώρησή
μας ότι οι a-priori πιθανότητες για κάθε κατηγορία (class priors) είναι
μεταξύ τους ίσες[^priori]. Επίσης, κάνουμε την υπόθεση ότι τα pixels δεδομένου μίας
κατηγορίας ακολουθούν από κοινού πολυδιάστατη κανονική με διαγώνιο και
σταθερό[^constant] πίνακα συνδιακιμάνσεων, γεγονός
που πιθανώς οδηγεί τον ταξινομητή σε κάποια από τα λανθασμένα αποτελέσματά του. 

[^priori]: Στο βήμα 14 υπολογίζουμε τις a-priori και βγαίνουν αρκετά διαφορετικές μεταξύ τους.

[^constant]: Όταν λέμε ότι ο πίνακας συνδιακιμάνσεων είναι σταθερός εννοούμε ότι είναι ακριβώς ο ίδιος για κάθε κατηγορία.

## Βήμα 12

Υλοποιούμε τον ευκλείδειο ταξινομητή ως έναν scikit-learn estimator (`class EuclideanDistanceClassifier`) ο οποίος υλοποιεί τις μεθόδους:

* fit (self, X , y) :
	μέθοδος που υπολογίζει την μέση τιμή των features για κάθε κλάση. Αυτό γίνεται με την χρήση της `digit_mean`.
	
* predict(self, X) : 
	μέθοδος που εκτιμά την κλάση του κάθε δείγματος υπολογίζοντας την ευκλείδεια απόσταση από τις μέσες τιμές των χατρακτηριστικών για κάθε κλάση. Αυτό το κάνει καλώντας την συνάρτηση `euclidean_distance_classifier`.
	
* score (self, X, y) :
	μέθοδος που υπολογίζει το accuracy του ταξινομητή με βάση τα labels (γνωστά από το σύνολο y)

Για αυτό το βήμα δεν παραθέτουμε κώδικα γιατί ουσιαστικά οι παραπάνω
συναρτήσεις αποτελούν wrappers των συναρτήσεων που έχουμε ήδη χρησιμοποιήσει.


## Βήμα 13

### Ερώτημα (α)

Για το βήμα αυτό υλοποιήσαμε την `evaluate_classifier` η οποία φαίνεται και παρακάτω.

```{.python}
def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    scores = cross_val_score(clf, X, y, 
                         cv=KFold(n_splits=folds, shuffle=True, random_state=42), 
                         scoring="accuracy")

    return np.mean(scores)
```

Όπως φαίνεται και στον κώδικα χρησιμοποιήσαμε την συνάρτηση `cross_val_score` για να υπολογίζουμε το accuracy με την μέθοδο του Cross Validation και χρησιμοποιώντας κατά προεπιλογή 5 folds.

Αυτό που κάνουμε στο `main.py` τελικά είναι να εκπαιδεύσουμε τον classifier με την κλήση της fit() και μετά να καλέσουμε την `evaluate_classifier`.

Στην έξοδο του προγράμματος βλέπουμε το εξής:

```{.html}
Euclidean Classifier Cross Validation score (accuracy): 84.995 %.
```

Άρα το accuracy με την μέθοδο του Cross Validation υπολογίστηκε ίσο με $84.995\%$.

<!-- TODO: check if cross validation score is usually higher that the simple score on one test set -->

### Ερώτημα (β)

Για τον σχεδιασμό της περιοχής απόφασης του ευκλείδειου ταξινομητή βασιστήκαμε στην συνάρτηση `plot_clf` που υπάρχει [εδώ](https://github.com/slp-ntua/prep-lab/blob/master/python/Lab%200.3%20Scikit-learn.ipynb).

Βέβαια ο ευκλείδειος ταξινομητής που εκπαιδεύσαμε δέχεται στην είσοδο του 256 διαφορετικά χαρακτηριστικά (ένα χαρακτηριστικό για κάθε pixel) ενώ εμείς θέλουμε να σχεδιάσουμε την περιοχή απόφασης στις δύο διαστάσεις.
Χρειαζόμαστε συνεπώς 2 μεγέθη συναρτήσει των οποίων θα γίνει ο σχεδιασμός. Τελικά επιλέξαμε τις δύο διευθύνσεις κατά μήκος των οποίων παρουσιάζεται η μεγαλύτερη διασπορά στα δεδομένα με την χρήση της PCA.

Παρακάτω φαίνεται η περιοχή απόφασης του ευκλείδειου ταξινομητή πάνω στο επίπεδο που ορίζεται από τις 2 πρώτες διευθύνσεις της PCA και περνά από την αρχή των αξόνων του χώρου των features (256D):

![Περιοχή Απόφασης του Ευκλείδειου Ταξινομητή](../figures/step_13_b_decision_plot.png)

Αρχικά θα ήταν λάθος να πούμε πως τα σημεία του scatterplot που βρίσκονται πάνω από περιοχή ίδιου χρώματος ότι ταξινομούνται ορθά και ότι τα σημεία που βρίσκονται πάνω από περιοχή άλλου χρώματος ταξινομούνται λανθασμένα. Αυτό ισχύει γιατί η περιοχή απόφασης του ταξινομητή υπολογίζεται πάνω στο επίπεδο που ορίζεται από τις δύο διευθύνσεις της PCA ενώ τα σημεία που φαίνονται αποτελούν προβολή των αρχικών δεδομένων στο επίπεδο αυτό. Παρά όλα αυτά μπορούμε να δούμε ότι πολλά από τα samples βρίσκονται πάνω από περιοχή ίδιου χρώματος επιβεβαιώνοντας ποιοτικά την ορθή λειτουργία του ταξινομητή.

Παρακάτω φαίνεται η τροποποιημένη `plot_clf`. Στην γραμμή 5 υπολογίζουμε την PCA και μετασχηματίζουμε τα δεδομένα στην γραμμή 7, τα οποία τελικά χρησιμοποιούνται στην γραμμή 34 για να παρασταθούν γραφικά. Επίσης στην γραμμή 24 χρησιμοποιούμε τον αντίστροφο μετασχηματισμό ώστε να μεταβούμε από τις 2 διαστάσεις στις 256 που χρησιμοποιεί ο ταξινομητής μας.

```{.python .numberLines}
def plot_clf(clf, X, y, labels, filename=""):


    #PCA to reduce to only 2 dimensions (initially we have 256 features)
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

```

### Ερώτημα (γ)

Στο βήμα αυτό σχεδιάσαμε την καμπύλη εκμάθησης του ευκλείδειου ταξινομητή χρησιμοποιώντας την έτοιμη συνάρτηση  `plot_learning_curve()` που βρίσκεται [εδώ](https://github.com/slp-ntua/prep-lab/blob/master/python/Lab%200.3%20Scikit-learn.ipynb). Η καμπύλη που σχεδιάσαμε φαίνεται παρακάτω και το score αποτελεί το accuracy του ταξινομητή:

![Καμπύλη εκμάθησης](../figures/step_13_c_learning_curves.png){width=90%}

Αρχικά παρατηρούμε ότι για πολύ μικρό πλήθος δειγμάτων εκπαίδευσης το train score είναι πολύ υψηλό. Αυτό συμβαίνει γιατί το πλήθος των δειγμάτων είναι αρκετά μικρό έτσι ώστε το μοντέλο του ευκλείδειου ταξινομητή να μπορεί να τα απομνημονεύσει. Το μικρό όμως πλήθος δειγμάτων εκπαίδευσης οδηγεί σε κακή γενίκευση του ταξινομητή σε νέα δείγματα και για αυτό παρατηρείται μικρό cross-validation score. Με λίγα λόγια έχουμε overfitting.

Καθώς το πλήθος των δειγμάτων εκπαίδευσης αυξάνεται το train score μειώνεται γιατί πλέον ο ταξινομητής δεν έχει την ικανότητα να απομνημονεύσει τα δείγματα. Ταυτόχρονα το cross-validation score αυξάνεται γιατί με την αύξηση των δειγμάτων εκπαίδευσης ταξινομητής μπορεί να γενικεύσει καλύτερα σε νέα δεδομένα.

Αυξάνοντας περαιτέρω το πλήθος των δειγμάτων εκπαίδευσης το training score και cross-validation score συγκλίνουν περίπου στην τιμή 85%. Η χρήση περισσότερων από περίπου 3000 δειγμάτων εκπαίδευσης δεν φαίνεται να βελτιώνει την επίδοση του ταξινομητή εφόσον η καμπύλη για το cross-validation score δεν φαίνεται να αυξάνεται. Ταυτόχρονα η τυπική απόκλιση του cross-validation score δεν φαίνεται να μικραίνει σημαντικά με την χρήση περισσότερων των 3000 δειγμάτων εκπαίδευσης. Συνεπώς η εκπαίδευση του ευκλείδειου ταξινομητή δεν έχει νόημα να γίνει σε training set με περισσότερα από 3000 δείγματα.



## Βήμα 14

Στο βήμα αυτό υλοποιήσαμε την συνάρτηση `calculate_priors` η οποία φαίνεται και παρακάτω.

```{.python}
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
```

Χρησιμοποιήσαμε την συνάρτηση αυτή για να υπολογίζουμε τις a-priori πιθανότητες για κάθε κλάση (κάθε ψηφίο 0-9) στο train set.

Στην έξοδο του προγράμματος βλέπουμε το εξής:

```{.html}
    P(digit_label=0) = 0.164
    P(digit_label=1) = 0.138
    P(digit_label=2) = 0.100
    P(digit_label=3) = 0.090
    P(digit_label=4) = 0.089
    P(digit_label=5) = 0.076
    P(digit_label=6) = 0.091
    P(digit_label=7) = 0.088
    P(digit_label=8) = 0.074
    P(digit_label=9) = 0.088
```
Παρατηρούμε ότι οι a-priori πιθανότητες των ψηφίων διαφοροποιούνται μεταξύ τους με εκείνες των ψηφίων 0-2 να είναι μεγαλύτερες από εκείνες των υπόλοιπων ψηφίων. 

<!--
Euclidean Classifier Cross Validation score (accuracy): 84.995 %.

```{.python include="../lab_1/Game_Of_Life.c" startLine=62 endLine=83 .numberLines}
```
-->


## Βήμα 15 

### Ερώτημα (α)

Ο Bayesian ταξινομητής χρησιμοποιεί για τις αποφάσεις του τις a posteriori πιθανότητες που σύμφωνα με τον κανόνα του Bayes δίνονται από τον τύπο:

$$P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)}$$

Για κάθε δείγμα x υπολογίζει για κάθε κλάση y την παραπάνω πιθανότητα και το κατατάσσει στην κλάση με την μεγαλύτερη a posteriori πιθανότητα. 
Δηλαδή θα κατατάξει το x στην κλάση

$$y^{*} = \underset{y}{argmax} {P(y|x)} = \underset{y}{argmax} \frac{P(x|y) \cdot P(y)}{P(x)} = \underset{y}{argmax} P(x|y) \cdot P(y)$$

Ο Naive Bayes Classifier υποθέτει ότι τα features κάθε δείγματος είναι μεταξύ τους ανεξάρτητα. Έτσι ο παραπάνω κανόνας απόφασης γίνεται: 

 $$y^{*} = \underset{y}{argmax} \prod_{i}{P(x_i|y)} \cdot P(y)$$

 Με τα $x_i$ να αποτελούν τα ανεξάρτητα features. Μάλιστα στην περίπτωση μας τα features αυτά είναι οι τιμές των pixel της εικόνας και υποθέτουμε ότι ακολουθούν κανονική κατανομή. Συνεπώς ισχύει ότι:

 $$P(x_i|y) = \mathcal{Ν}(x_i;\mu_{i,y},\sigma_{i,y}) \qquad i=0,...,255$$

Υλοποιούμε τον Naive Bayesian Classifier ως έναν scikit-learn estimator (`class CustomNBClassifier`) που υλοποιεί τις μεθόδους fit, predict και score.

Τις μέσες τιμές $μ_{i,y}$ και τις τιμές διασποράς $\sigma_{i,y}$[^not_calc_if_unit_var] τις υπολογίσαμε όπως και στα προηγούμενα ερωτήματα στην συνάρτηση `fit()`. Όμως σε κάποια pixels η τιμή της διασποράς είναι μηδενική (που δημιουργεί προβλήματα διαίρεσης με το 0). Για αυτό εφαρμόζουμε το λεγόμενο variance smoothing προσθέτοντας μία πολύ μικρή σταθερά στην τιμή της διασποράς σε κάθε pixel. Επιλέξαμε να προσθέσουμε την τιμή $10^-5$ καθώς για μικρότερες τιμές προέκυπτε overflow. Επίσης στο τέλος της συνάρτησης fit υπολογίζονται και οι a priori πιθανότητες, δηλαδή οι $P(y=k) \; k=0,..,9$, με την χρήση της συνάρτησης `calculate_priors` του ερωτήματος 14.

Για την συνάρτηση `predict` διατρέχουμε κάθε sample το οποίο της δίνεται ως είσοδος.
Για κάθε sample υπολογίζουμε την τιμή $P(x|y) \cdot P(y)$ για κάθε κλάση και ταξινομούμαι βάση με την χρήση του `argmax`. Η τιμή $P(x|y)$ υπολογίζεται με μία
βοηθητική συνάρτηση της κλάσης που την ονομάσαμε `direct_prob` και υπολογίζει το
γινόμενο των $P(x_i|y)$. Παρακάτω φαίνεται ο κώδικας της `predict` και της `direct_prob`:

```{.python}
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
            pred = np.argmax(direct * self.priors);

            predictions.append(pred)


        return predictions

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
```

Τελικά για το ερώτημα αυτό απλά φτιάχνουμε ένα instance του `CustomNBClassifier`, καλούμε την `fit` πάνω στο train set. Δεν καλούμε την `predict` γιατί μπορούμε να υπολογίσουμε στο ερώτημα (β) κατευθείαν το accuracy καλώντας την `score`. Αυτό το κάνουμε στο `main.py`.


[^not_calc_if_unit_var]: Σε περίπτωση που `use_unit_variance == Τrue` δεν υπολογίζουμε τις διασπορές και τις θέτουμε απλά ίσες με 1.

### Ερώτημα (β)

Υπολογίζουμε την ακρίβεια που πετυχαίνουμε και στην έξοδο του προγράμματος βλέπουμε το εξής:

```{.html}
Custom naive Bayes test accuracy = 74.73 %.
```

Άρα το accuracy στο test set υπολογίστηκε ίσο με $74.73\%$ όταν η εκπαίδευση γίνεται με βάση την δική μας υλοποίηση του Naive Bayes ταξινομητή. 

### Ερώτημα (γ)

Εκπαιδεύουμε τον Naive Bayesian Classifier με χρήση της έτοιμης υλοποίησης `GaussianNB()` της βιβλιοθήκης `scikit-learn` για να ταξινομήσουμε τα ψηφία του test set. Αυτός ο ταξινομητής έχει την ίδια υλοποίηση με την δικό μας ταξινομητή. Μάλιστα θέτουμε ως παράμετρο var_smoothing την τιμή $10^{-5}$ που έχουμε χρησιμοποιήσει και στην δική μας υλοποίηση ώστε να μπορούμε να συγκρίνουμε καλύτερα τα αποτελέσματα. 

Υπολογίζουμε την ακρίβεια που πετυχαίνουμε και στην έξοδο του προγράμματος βλέπουμε το εξής:

```{.html}
Custom naive Bayes test accuracy = 74.73 %.
sklearn naive Bayes test accuracy = 74.68 %.
```

Άρα το accuracy στο test set υπολογίστηκε ίσο με $74.68\%$ όταν χρησιμοποιείται ο Naive Bayes ταξινομητής της βιβλιοθήκης `scikit-learn`. Όσον αφορά την σύγκριση της δικής μας υλοποίησης με εκείνη της `scikit-learn` το accuracy που πετυχαίνουμε είναι σχεδόν ίδιο. Παρατηρούμε ότι το accuracy που πετυχαίνουμε είναι υποδεέστερο συγκριτικά με το accuracy ( $81.42\%$ ) στον Ευκλείδειο ταξινομητή.

## Βήμα 16

Στο βήμα αυτό θεωρούμε ότι όλα τα pixel έχουν μοναδιαία διασπορά, δηλαδή ότι ο πίνακας συνδιακύμανσης για όλα τα χαρακτηριστικά είναι μοναδιαίος ($Σ = Ι$). Επειδή ο Naive Bayesian ταξινομητής θεωρεί ότι όλα τα δείγματα ακολουθούν κανονική κατανομή με την υπόθεση μοναδιαίας διασποράς ουσιαστικά πρόκειται για τον Euclidean Distance ταξινομητή, μόνο που εδώ δεν θεωρούμε όλες τις a priori πιθανότητες ίσες μεταξύ τους. 

Εκπαιδεύουμε τον ταξινομητή και υπολογίζουμε το accuracy που επιτυγχάνεται στο test set. 
Στην έξοδο του προγράμματος βλέπουμε το εξής:

```{.html}
Custom naive Bayes with unit variance test accuracy = 81.26 %
```

Άρα το accuracy στο test set υπολογίστηκε όσο με $81.26\%$ που είναι σχεδόν ίδιο με εκείνο που είχαμε πετύχει με την χρήση του Ευκλείδειου ταξινομητή ($81.42\%$).
Παρατηρούμε ότι το αποτέλεσμα υποθέτοντας μοναδιαία διασπορά για όλες τα δείγματα είναι αρκετά καλύτερο από εκείνο που πέτυχε ο Naive Bayesian ταξινομητής που είδαμε στο Ερώτημα 15. 

**Γιατί έχουμε καλύτερη επίδοση με μοναδιαία διασπορά;**

Αρχικά παρατηρήσαμε ότι η διασπορά που υπολογίζεται για μερικά pixels
είναι πολύ μικρή. Αυτό φαίνεται κιόλας στο βήμα 8 όπου ένα μεγάλο ποσοστό
των pixels έχει μηδενική διασπορά.

Έστω ότι το pixel $i$ και για την κατηγορία $y_0$ έχει υπολογιστεί η μέση τιμή
$μ_{i,y_0}$ και η διασπορά $σ_{i,y_0} \ll 1$. Αυτό σημαίνει ότι η $P(x_i|y_0)$
παίρνει μεγάλες τιμές μόνο σε μία πολύ μικρή περιοχή γύρω από το $μ_{i,y_0}$
ενώ για τις υπόλοιπες τιμές του $x_i$ η πιθανότητα είναι πρακτικά μηδενική.
Ως αποτέλεσμα αν το pixel $i$ μίας εικόνας έχει έστω και ελάχιστα διαφορετική
τιμή από $μ_{i,y_0}$, τότε ισχύει ότι:

$$P(x|y_0) = \prod_{j=0}^{255}{P(x_j|y_0)} \approx 0$$

Καταλαβαίνουμε συνεπώς ότι μία εικόνα που όντως ανήκει στην κατηγορία $y_0$
μπορεί πολύ εύκολα να μην ταξινομηθεί σε αυτήν την κατηγορία γιατί κάποιο pixel
είναι λίγο διαφορετικό από την μέση τιμή. Από την άλλη όταν χρησιμοποιούμε
μοναδιαία διασπορά για όλα τα pixels δεν έχουμε αυτήν την ακραία συμπεριφορά.
Αυτός είναι και ο λόγος που έχουμε μεγαλύτερο accuracy
στην περίπτωση όπου χρησιμοποιούμε την μοναδιαία διασπορά.

Αυτό το φαινόμενο μοιάζει πολύ με overfitting στην περίπτωση όπου
υπολογίζουμε την μέση τιμή αφού το μοντέλο ταιριάζει πάρα πολύ έντονα στα
train δεδομένα, είναι ευαίσθητο σε μικρές μεταβολές της εισόδου και προφανώς
δεν γενικεύει καλά (έχει μικρό accuracy στο test set).



## Βήμα 17

Σε αυτό το βήμα υλοποιήσαμε τους σκελετούς των συναρτήσεων που μας δίνονται
στο αρχείο `lib.py`. Δεν θα σχολιάσουμε ιδιαίτερα τον κώδικα γιατί απλά
φτιάχναμε ένα instance του ταξινομητή που θέλαμε να μελετήσουμε και υπολογίζαμε το accuracy του με την χρήση της cross-validation μεθόδου.
Ενδεικτικά παραθέτουμε παρακάτω μία από αυτές τις συναρτήσεις:

```{.python}
def evaluate_linear_svm_classifier(X, y, folds=5):
    """Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """

    clf = SVC(kernel="linear", probability=True)

    score = evaluate_classifier(
        clf,
        X, 
        y,
        folds=5
    )
    
    name = "SVM linear"
    print(f"{name: <20} | {score*100:.3f} %")
```

Από την έξοδο του προγράμματος λαμβάνουμε τον εξής πίνακα ταξινομημένος
με φθίνουσα σειρά ως προς το accuracy:


|  Model Name          | Accuracy |
|        :---          |  :---    |
| SVM poly             | 98.121 % |
| SVM rbf              | 97.696 % |
| 1-NN                 | 97.079 % |
| 5-NN                 | 96.132 % |
| SVM linear           | 95.776 % |
| 15-NN                | 94.843 % |
| 51-NN                | 91.592 % |
| SVM sigmoid          | 89.233 % |
| Custom NB unit var   | 85.050 % |
| sklearn Naive Bayes  | 78.124 % |
| Custom NB            | 78.178 % |



O ταξινομητής με την καλύτερη επίδοση είναι αυτός που χρησιμοποιεί
το SVM με το πολυωνυμικό kernel. Μετά ακολουθεί αυτό που χρησιμοποιεί
το radial basis function kernel και ο ταξινομητής 1-NN.

Σχετικά με τους k-NN ταξινομητή παρατηρούμε ότι το accuracy μειώνεται
καθώς αυξάνομε το $k$. Συνεπώς η βέλτιστη τιμή για το k σε περίπτωση
που χρησιμοποιούσαμε έναν k-NN ταξινομητή είναι $k=1$.

Σχετικά με τα Naive Bayes μοντέλα παρατηρούμε ότι σχολιάσαμε και στα
προηγούμενα βήματα. Αρχικά υπάρχει πάρα πολύ μικρή διαφορά μεταξύ της
δικής μας υλοποίησης και της υλοποίησης της του sklearn στην περίπτωση
όπου δεν χρησιμοποιούμε μοναδιαίες διασπορές. Επίσης το μοντέλο Naive
Bayes που χρησιμοποιεί μοναδιαίες διασπορές έχει $7\%$ μεγαλύτερο cross-validation
accuracy σε σχέση με τα άλλα.




## Βήμα 18

Στο βήμα αυτό δοκιμάζουμε δύο τεχνικές ensembling: τον Voting και τον Bagging Classifier. 

### Ερώτημα (α)

Στο ερώτημα αυτό δοκιμάζουμε να χρησιμοποιήσουμε έναν Voting Classifier. Θα εξετάσουμε δύο τρόπους ψηφοφορίας που μπορεί να χρησιμοποιήσει ένας Voting Classifier για την ταξινόμηση: hard και soft.

Όσον αφορά τον πρώτο τρόπο, ο ταξινομητής κατατάσσει κάθε δείγμα εισόδου στην κλάση στην οποία το ταξινομεί η πλειοψηφία των επιμέρους ταξινομητών. Σε περίπτωση που το dataset αφορά πρόβλημα όπου έχουμε δύο μόνο κλάσεις επιλέγεται μονός αριθμός ταξινομητών ώστε να αποφευχθούν σενάρια ισοπαλίας. Όταν οι κλάσεις είναι περισσότερες, όπως συμβαίνει και στο δικό μας dataset (έχουμε 10 κλάσεις), o μονός αριθμός ταξινομητών δεν μπορεί να μας εξασφαλίσει ότι δεν θα προκύψει κάποια ισοπαλία. Για παράδειγμα αν έχουμε
11 ταξινομητές μπορεί οι 5 πρώτοι να ταξινομούν στο ψηφίο 0, οι 5 επόμενοι στο ψηφίο 1 και ο τελευταίος στο ψηφίο 2. Βέβαια αν έχουμε μονό αριθμό ταξινομητών αποφεύγεται η ισοπαλία στην περίπτωση όπου οι μισοί υποστηρίζουν την κλάση Α ενώ οι άλλοι μισή την κλάση Β.

Με την soft voting μέθοδο, υπολογίζεται το άθροισμα των πιθανοτήτων των επιμέρους ταξινομητών και το κάθε δείγμα ταξινομείται στην κλάση στην οποία αντιστοιχεί το μεγαλύτερο άθροισμα. Καθώς η ταξινόμηση εδώ γίνεται με χρήση των αθροισμάτων, ο μονός αριθμός επιμέρους ταξινομητών δεν μπορεί να αποκλείσει σενάρια ισοπαλίας (ούτε όταν έχουμε 2 μόνο κλάσεις). Βέβαια στην πράξη η ισοπαλία με χρήση soft voting δεν συμβαίνει σχεδόν ποτέ.

Προκειμένου να βελτιώσουμε την ακρίβεια που μπορούμε να πετύχουμε συνδυάζουμε ταξινομητές που κάνουν διαφορετικό τύπο λαθών. Στην περίπτωσή μας αυτό σημαίνει ότι τείνουν να ταξινομούν λανθασμένα (τους "μπερδεύουν") περισσότερο διαφορετικά ψηφία. Επομένως για να μπορέσουμε να διαλέξουμε ποιοι συνδυασμοί ταξινομητών μπορούν να οδηγήσουν σε μεγαλύτερο accuracy κρίνουμε χρήσιμη την διάταξη για κάθε ταξινομητή των ψηφίων με φθίνουσα σειρά πλήθους δειγμάτων που κάνει (τα ψηφία) mispredict. 

Για τον σκοπό αυτό υλοποιούμε στην `lib.py` την συνάρτηση `mispredictions` που φαίνεται παρακάτω.

```{.python}
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
 ```

 Από την έξοδο του προγράμματος λαμβάνουμε τον εξής πίνακα:
 
| Classifier           | Digits in order of decreasing misprediction rate |
| :----------------:   | :-----------------------------------------------:|
| Gaussian Naive Bayes | [4 5 3 0 2 8 9 7 6 1]                            |
| 1-NN                 | [8 4 5 2 3 1 9 7 6 0]                            |
| 5-NN                 | [4 5 2 8 3 9 7 6 1 0]                            |
| 15-NN                | [8 4 2 5 3 7 6 9 1 0]                            |
| 51-NN                | [2 4 8 5 3 6 9 7 0 1]                            |
| SVM linear kernel    | [5 8 4 3 2 7 6 1 0 9]                            |
| SVM poly kernel      | [3 2 4 8 5 7 6 9 1 0]                            |
| SVM rbf kernel       | [3 2 4 8 6 1 7 5 9 0]                            |
| SVM sigmoid kernel   | [2 5 8 0 3 4 6 7 9 1]                            |

Για κάθε ταξινομητή όσο αριστερότερα βρίσκεται ένα ψηφίο στην αντίστοιχη λίστα τόσο λιγότερο καλός είναι στο να το ταξινομήσει σωστά. Για παράδειγμα στον Gaussian Naive Bayes το ψηφίο
4 εμφανίζεται τέρμα αριστερά, που σημαίνει ότι ο ταξινομητής αυτός κάνει τα περισσότερα λάθη
ταξινόμησης όταν το test sample ανήκει στην κατηγορία 4.

Από τον παραπάνω πίνακα παρατηρούμε ότι οι SVM rbf και poly kernel είναι αρκετά παρόμοιοι όσον αφορά τα λάθη που κάνουν στην κατηγοριοποίηση των ψηφίων. Επιλέγουμε έναν από αυτούς, εκείνον που πέτυχε την μεγαλύτερη ακρίβεια, δηλαδή τον SVM poly kernel. Ο ταξινομητής αυτός τείνει να ταξινομεί τα ψηφία 2, 3 αρκετές φορές λανθασμένα. Για αυτό θα τον συνδυάσουμε με τον 1-ΝΝ ταξινομητή που ταξινομεί τα ψηφία αυτά ορθώς σχετικά συχνά. Σε έναν Voting Classifier μπορούμε να θεωρήσουμε βάρη για την στάθμιση της επιρροής που έχει κάθε ταξινομητής στο τελικό αποτέλεσμα. Συνδυάζοντας τους δύο αυτούς ταξινομητές με βάρη 8 και 7 για τον SVM poly kernel και τον 1-NN αντίστοιχα παίρνουμε στην έξοδο του προγράμματος:

```{.html}
Voting classifier using classifiers: SVM poly kernel, 1-NN, hard voting method has accuracy of 98.121 %.
Voting classifier using classifiers: SVM poly kernel, 1-NN, soft voting method has accuracy of 98.148 %.
```

Επομένως, το accuracy χρησιμοποιώντας soft voting είναι ελαφρώς υψηλότερο σε σχέση με τον SVM poly kernel, ο οποίος πετυχαίνει accuracy = 98.121 % (τόσο πετυχαίνουμε με χρήση hard voting). Διαφορετικοί συνδυασμοί βαρών που δίνουν περισσότερο βάρος στον SVM poly kernel πετυχαίνουν μικρότερη ακρίβεια (soft voting method). 

Δοκιμάζουμε να συνδυάσουμε τους παραπάνω με έναν ακόμα ταξινομητή. Από τους υπόλοιπους ταξινομητές δεν μπορούμε να διακρίνουμε κάποιους που κάνουν αξιοσημείωτα διαφορετικά λάθη από αυτούς που έχουμε διαλέξει ήδη. Ως 3ο ταξινομητή επιλέγουμε τον SVM linear kernel καθώς έχει καλό accuracy σε σχέση με αυτούς που έχουν μείνει και κάνει λάθη κυρίως στο 5 (το οποίο δεν μπερδεύει σημαντικά τον SVM poly kernel ούτε αποτελεί το κύριο σημείο λαθών του 1-ΝΝ). Το 8 το μπερδεύουν αρκετά και ο 1-NN και ο SVM linear kernel (και το 4 το μπερδεύουν όλοι σε σημαντικό βαθμό) αλλά δεδομένων των διαθέσιμων ταξινομητών είναι μάλλον μία καλή επιλογή. Θα δώσουμε μεγαλύτερο βάρος στους ταξινομητές με μεγαλύτερο accuracy. Έτσι, καθώς ο SVM poly kernel είναι ο "καλύτερος" θα θέσουμε το μεγαλύτερο βάρος σε αυτόν. 

Συνοψίζουμε τα στοιχεία των ταξινομητών που συνδυάζουμε και τα βάρη που δίνουμε στον καθένα στον παρακάτω πίνακα.

| Classifier           | Digits in order of decreasing misprediction rate | Accuracy | Weights |
| :----------------:   | :-----------------------------------------------:   |  :---:   |  :---:   |
| SVM poly kernel      | [3 2 4 8 5 7 6 9 1 0]                               | 98.121%  |    8    |
| 1-NN                 | [8 4 5 2 3 1 9 7 6 0]                               | 97.079%  |    2    |
| SVM linear kernel    | [5 8 4 3 2 7 6 1 0 9]                               | 97.696%  |    1    |

Στην έξοδο του προγράμματος παίρνουμε:

```{.html}
Voting classifier using classifiers: SVM poly kernel, 1-NN, SVM linear kernel, hard voting method has accuracy of 98.121 %.
Voting classifier using classifiers: SVM poly kernel, 1-NN, SVM linear kernel, soft voting method has accuracy of 98.203 %.
```
Επομένως, ο voting classifier που συνδυάζει τον 1-ΝΝ, SVM linear kernel και τον SVM poly kernel έχει accuracy $98.121 \%$ και $98.203 \%$ (soft και hard voting method αντίστοιχα). Παρατηρούμε ότι τα αποτελέσματα είναι ικανοποιητικά. To accuracy με hard voting είναι ακριβώς το ίδιο με εκείνο του SVM poly kernel και με soft voting είναι ελαφρώς υψηλότερο σε σχέση με τον SVM poly kernel, ο οποίος πετυχαίνει accuracy = $98.121 \%$.

Δοκιμάζουμε στην συνέχεια τον 5-NN αντί για τον SVM linear kernel (έχει μεγαλύτερο accuracy). Δηλαδή τους ταξινομητές:

| Classifier           | Digits in order of decreasing misprediction rate | Accuracy | Weights |
| :----------------:   | :-----------------------------------------------:|  :---:   |  :---:  |
| SVM poly kernel      | [3 2 4 8 5 7 6 9 1 0]                            | 98.121%  |    8    |
| 1-NN                 | [8 4 5 2 3 1 9 7 6 0]                            | 97.079%  |    1    |
| 5-NN                 | [4 5 2 8 3 9 7 6 1 0]                            | 96.132%  |    2    |

Στην έξοδο του προγράμματος έχουμε:

```{.html}
Voting classifier using classifiers: SVM poly kernel, 5-NN, 1-NN, hard voting method has accuracy of 98.121 %.
Voting classifier using classifiers: SVM poly kernel, 5-NN, 1-NN, soft voting method has accuracy of 98.217 %.
```

Επομένως, ο voting classifier αυτός έχει accuracy $98.121 \%$ και $98.217\%$ (soft και hard voting method αντίστοιχα), δηλαδή έχουμε μία μικρή βελτίωση σε σχέση με πριν χρησιμοποιώντας soft voting. 
<!--
Δοκιμάζουμε τώρα να συνδυάζουμε όλους τους παραπάνω, δηλαδή τους ταξινομητές:
| Classifier           | Digits in order of decreasing misprediction rate | Accuracy | Weights |
| :----------------:   | :-----------------------------------------------:|  :---    |  :---   |
| SVM poly kernel      | [3 2 4 8 5 7 6 9 1 0]                            | 98.121 % |   10    |
| SVM linear kernel    | [5 8 4 3 2 7 6 1 0 9]                            | 97.696 % |    1    |
| 1-NN                 | [8 4 5 2 3 1 9 7 6 0]                            | 97.079 % |    1    |
| 5-NN                 | [4 5 2 8 3 9 7 6 1 0]                            | 96.132 % |    1    |

Στην έξοδο του προγράμματος έχουμε:
```

```
Πετύχαμε το υψηλότερο accuracy μέχρι τώρα, δηλαδή , με soft voting.
-->

Τέλος, δοκιμάζουμε να συνδυάσουμε τους 3 ταξινομητές που ξεχωριστά πετυχαίνουν τα 3 υψηλότερα accuracy, δηλαδή τους SVM poly kernel, SVM rbf kernel και 1-NN με βάρη 8, 4, και 2 αντίστοιχα.

Στην έξοδο του προγράμματος έχουμε:

```{.html}
Voting classifier using classifiers: SVM poly kernel, SVM rbf kernel, 1-NN, hard voting methodhas accuracy of 98.121 %.
Voting classifier using classifiers: SVM poly kernel, SVM rbf kernel, 1-NN, soft voting method has accuracy of 98.121 %.
```

To accuracy που πετυχαίνουμε είναι αρκετά καλό αλλά ίδιο με εκείνο του SVM poly kernel και με τις δύο μεθόδους voting. 

Σε όλα τα παραπάνω πειράματα για την μέτρηση της ακρίβειας χρησιμοποιείται 5-fold cross-validation. Να σημειωθεί ότι καθώς τα υποσύνολα που επιλέγονται κάθε φορά είναι τυχαία μπορεί η τιμή του accuracy που παίρνουμε να διαφοροποιείται ελαφρώς στην έξοδο του προγράμματος από αυτή που αναγράφεται στην αναφορά.

### Ερώτημα (β)

Στο ερώτημα αυτό δεν υλοποιήσαμε τον δικό μας BaggingClassifier. Χρησιμοποιήσαμε την έτοιμη υλοποίηση του ταξινομητή της `scikit-learn`.

Ένας Bagging Classifier χωρίζει το training set σε τυχαία υποσύνολα και χρησιμοποιεί κάθε ένα από αυτά για να εκπαιδεύσει έναν ταξινομητή βάσης. Αυτό γίνεται με σκοπό οι classifiers να διαφοροποιούνται μεταξύ τους και να έχουν διαφορετικό τύπο λαθών. Τελικά κάθε δείγμα ταξινομείται κατόπιν ψηφοφορίας ή μέσου όρου επί των επιμέρους προβλέψεων ώστε να επιτύχουμε καλύτερη επίδοση.

Αρχικά χρησιμοποιούμε ως ταξινομητή βάσης τον **SVM poly kernel** καθώς αυτός είναι σημείωσε το μεγαλύτερο accuracy. Επιλέξαμε να χρησιμοποιήσουμε τυχαία υποσύνολα με μέγεθος όσο το $80 \%$ του train set για την εκπαίδευση των ταξινομητών, με ενεργοποιημένο το bootstrapping. Πειραματιζόμαστε με 5, 10, 15 και 25 ταξινομητές και στην έξοδο του προγράμματος έχουμε:

```{.html}
Bagging classifier with 5 base estimators has accuracy of 97.888 %
Bagging classifier with 10 base estimators has accuracy of 97.902 %
Bagging classifier with 15 base estimators has accuracy of 97.929 %
Bagging classifier with 25 base estimators has accuracy of 97.956 %
```

Επομένως, δεν έχουμε κάποια βελτίωση του accuracy σε σχέση με τον μεμονωμένο ταξινομητή SVM poly kernel που είχε accuracy $98.12 \%$. Παρατηρούμε ότι το accuracy αυξάνεται δυσανάλογα λίγο σε σχέση με την αύξηση του πλήθους των ταξινομητών που χρησιμοποιούνται στο Bagging. Οπότε δεν περιμένουμε να ξεπεράσουμε το accuracy του $98.12 \%$ με έναν "λογικό" αριθμό από classifiers.

Βέβαια, ο ταξινομητής SVM poly kernel έχει ήδη πολύ μεγάλο accuracy επομένως υπάρχει μικρό περιθώριο για περαιτέρω αύξηση. Για αυτό εξετάζουμε και το **SVM με linear kernel** που είναι ένας ταξινομητής με κάπως μικρότερο accuracy.  Στην έξοδο του προγράμματος παίρνουμε τα εξής αποτελέσματα:

```{.html}
Bagging classifier with 5 base estimators has accuracy of 96.461 %
Bagging classifier with 10 base estimators has accuracy of 96.544 %
Bagging classifier with 15 base estimators has accuracy of 96.544 %
Bagging classifier with 25 base estimators has accuracy of 96.599 %
```

 Ο μεμονωμένος ταξινομητής **SVM linear kernel** έχει accuracy $95.77 \%$ και το χαμηλότερο accuracy που πετύχαμε με bagging είναι $96.46 \%$. Επομένως, με χρήση του bagging ταξινομητή καταφέραμε να βελτιώσουμε το accuracy. Το accuracy αυξάνεται ελαφρώς όσο αυξάνουμε το πλήθος των ταξινομητών που χρησιμοποιεί ο bagging classifier. Παρόλα αυτά η βελτίωση αυτή είναι αρκετά περιορισμένη δεν κρίνουμε σκόπιμο να αυξήσουμε περαιτέρω το πλήθος των estimator.

### Ερώτημα (γ)

Και οι δύο μέθοδοι στοχεύουν στο να αυξήσουν την ικανότητα γενίκευσης
συνδυάζοντας πολλούς ταξινομητές μαζί. Στην περίπτωση του Voting Classifier
συνδυάζουμε διαφορετικούς ταξινομητές μεταξύ τους που όλοι εκπαιδεύονται σε όλο
το train set, ενώ στον Bagging Classifier χρησιμοποιούμαι το ίδιο μοντέλο
πολλές φορές αλλά σε διαφορετικά τυχαία υποσύνολα του train set.


**Σχετικά με το Voting Classifier**

Όσον αφορά τον voting classifier, η βελτίωση που πετυχαίνουμε δεν είναι ιδιαίτερα ικανοποιητική. Καταφέραμε βελτίωση του accuracy που πετυχαίνει ο μεμονωμένος καλύτερος ταξινομητής μας (ο SVM poly kernel) της τάξης του σημαντικότερου δεκαδικού ψηφίου. Χρησιμοποιώντας hard voting μειώθηκε η ακρίβεια του SVM poly kernel πιθανώς επειδή πλειοψηφία των ταξινομητών που συνδυάζουμε "παρέσυρε" τον voting classifier σε λάθος αποφάσεις για κάποια δείγματα. Επιλέγοντας soft voting (και σταθμίζοντας με βάρη κάθε επιμέρους ταξινομητή) αυξάνουμε την "ανοσία" του voting ταξινομητή σε λάθος αποφάσεις των επιμέρους ταξινομητών που έχουν μικρότερη ακρίβεια. 

Τελικά όμως, αν και κατά ελάχιστο, καταφέραμε με τον συνδυασμό των SVM poly kernel, 5­NN, 1­NN με soft voting και τα κατάλληλα βάρη να πετύχουμε το μεγαλύτερο accuracy μέχρι στιγμής
που είναι ίσο με $98.217 \%$.

<!-- TODO: checking needed 
(info from : https://www.math.univ-toulouse.fr/~agarivie/Telecom/apprentissage/articles/BaggingML.pdf )
-->

**Σχετικά με το Bagging Classifier**

<!--
Γενικά, ένας bagging classifier καθώς παίρνει τον μέσο των προβλέψεων που προκύπτουν από την εκπαίδευση του ταξινομητή βάσης σε τυχαία υποσύνολα του train set προσπαθεί να αυξήσει το "robustness" ενάντια σε outliers μειώνοντας την επίδρασή τους. Η απουσία σημείων με αρχικά μεγάλη επιρροή μειώνει την επιρροή τους για να αυξήσει την σταθερότητα του μοντέλου και έτσι να αυξήσει το accuracy. Σημαντικό εδώ είναι το bootstrap sampling. 
-->
Με το bagging classifier δεν καταφέραμε να κατασκευάσουμε
κάποιον ταξινομητή με το μέγιστο accuracy μέχρι στιγμής.
Καταφέραμε όμως να αυξήσουμε το accuracy σε σχέση με τα μεμονωμένα
μοντέλα αρκετά περισσότερο από ότι με το Voting Classifier.
Στην περίπτωσή μας, η χρήση του bagging ταξινομητή καταφέρνει να βελτιώσει κατά
σχεδόν $1\%$ το accuracy του SVM linear kernel. Όμως όταν ως ταξινομητής βάσης
χρησιμοποιείται ο SVM poly kernel που εξαρχής έχει αρκετά υψηλό accuracy, το
accuracy που πετυχαίνει είναι υποδεέστερο. 

## Βήμα 19

Αρχικά σε αυτό το βήμα γράψαμε κατευθείαν τον κώδικα που χρειάστηκε
στην κλάση `PytorchNNModel` που είναι συμβατή με το sklearn.
Επίσης προσθέσαμε στο `lib.py` την κλάση `FullyConnectedNeuralNetwork`
που υλοποιεί το fully connected neural network σε PyTorch.

Συνεπώς ουσιαστικά είναι σαν να έχουμε ακολουθήσει τα ερωτήματα με την
σειρά (γ), (α), (β), (δ).


### Ερώτημα (α)

Σε αυτό το ερώτημα δεν υλοποιήσαμε κάποια δικιά μας κλάση Dataloader.
Χρησιμοποιήσαμε τις ήδη έτοιμες κλάσεις `TensorDataset` και `DataLoader`.
Την πρώτη από αυτές την χρησιμοποιήσαμε ώστε να 'μετατρέψουμε' τα δεδομένα
μας που έχουμε ήδη σε numpy arrays αρχικά σε torch tensors και μετά σε
ένα Dataset του pytorch.

Ως minibatch size επιλέξαμε το 16. Δεν πειραματιστήκαμε παρακάτω με αυτήν
την τιμή και την κρατήσαμε σταθερή.

Ως παράδειγμα παρακάτω φαίνεται τμήμα του κώδικα για το ερώτημα (γ) που
χρησιμοποιούμε τα dataloaders:

```{.python}
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
```

### Ερώτημα (β)

#### Ο κώδικας

Για αυτό το ερώτημα υλοποιήσαμε την κλάση `FullyConnectedNeuralNetwork`.
Αυτή η κλάση παίρνει ως παράμετρο το `LAYER_SIZE` που είναι μία python
λίστα που περιγράφει το μέγεθος του κάθε επιπέδου. Μάλιστα το πρώτο
στοιχείο της λίστας αυτής ορίζει το μέγεθος της εισόδου ενώ το τελευταίο
το μέγεθος της εξόδου του νευρωνικού δικτύου. Επίσης μεταξύ
των γραμμικών επιπέδων του δικτύου προσθέτουμε και relu επίπεδα.
Για παράδειγμα η λίστα `LAYER_SIZE = [256, 100, 10]` παράγει το παρακάτω δίκτυο:

```{.html}
FullyConnectedNeuralNetwork(
  (layers): ModuleList(
    (0): Linear(in_features=256, out_features=100, bias=True)
    (1): ReLU()
    (2): Linear(in_features=100, out_features=10, bias=True)
  )
)
```

Παρακάτω φαίνεται ο constructor της κλάσης που δημιουργεί το μοντέλο
με την χρήση ενός for loop πάνω στην παράμετρο `LAYER_SIZE`

```{.python}
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
```

Επίσης το forawrd pass στο δίκτυο αυτό είναι αρκετά απλό. Απλά εφαρμόζουμε
με ένα for loop το κάθε επίπεδο στην είσοδο. Πρέπει να σημειώσουμε ότι
η κλάση αυτή επιστρέφει logits, δηλαδή δεν επιστρέφει πιθανότητες αφού
δεν έχουμε στο τέλος ένα softmax επίπεδο. Αυτό έχει γίνει γιατί αργότερα
θα χρησιμοποιήσουμε το `CrossEntropyLoss` για την εκπαίδευση το οποίο
αναμένει logits στην είσοδο του αφού λαμβάνει από μόνο του υπόψη το softmax.
Παρακάτω φαίνεται η συνάρτηση `forward` του νευρωνικού δικτύου:

```{.python}
def forward(self, x):

        for layer in self.layers:
            # we cast to float because it otherwise throws
            # an error. (x is of type double without casting)
            x = layer(x.float())

        logits = x

        return logits
```

#### Πειραματισμός με το νευρωνικό δίκτυο {#nn-acc_commnets}

Δοκιμάσαμε την επίδοση των εξής περιπτώσεων:

| LAYER_SIZE                    | Πλήθος Εποχών |
|  :---                         | :---:         |
| [256, 10]                     | 10            |
| [256, 100, 10]                | 10            |
| [256, 100, 100, 10]           | 20            |
| [256, 1000, 10]               | 20            |

και τα αποτελέσματα είναι τα εξής:

| Layers              | Εποχές | Πλήθος Παραμέτρων | Test Set Accuracy |
| :---                | :---:  | :---:             | :---:             |
| [256, 10]           | 10     | 2570              | 88.290 %          |
| [256, 100, 10]      | 10     | 26710             | 90.732 %          |
| [256, 100, 100, 10] | 20     | 36810             | 91.280 %          |
| [256, 1000, 10]     | 20     | 267010            | 92.027 %          |

Παρατηρούμε συνεπώς ότι με την αύξηση του πλήθους επιπέδων αλλά και με
την αύξηση των νευρώνων ανά επίπεδο επιτυγχάνουμε καλύτερη επίδοση.
Γενικότερα παρατηρούμε ότι με την αύξηση του πλήθους των παραμέτρων
του δικτύου αυξάνεται και το accuracy. Βέβαια με την αύξηση του πλήθους
των παραμέτρων αυξάνεται και ο χρόνος εκπαίδευσης, το οποίο ήταν αισθητό.
Επίσης παρατηρούμε ότι το accuracy αυξάνεται δυσανάλογα λιγότερο από
την αύξηση του πλήθους των παραμέτρων. 

<!-- NN accuracy in on test set, SVM cross-validation
Επίσης παρατηρούμε ότι τα νευρωνικά δίκτυα
δεν φτάσανε το accuracy των SVM και συγκεκριμένα αυτών που χρησιμοποιούν poly, rbf και linear kernel.(Είχαν accuracy $98.12$ $97.69$ και $95.77$ αντίστοιχα.) Θα μπορούσαμε ίσως να δοκιμάζαμε να κατασκευάζαμε ένα αρκετά μεγαλύτερο νευρωνικό δίκτυο με αρκετά επίπεδα και αρκετούς νευρώνες ανά επίπεδο ώστε να πετύχουμε μεγαλύτερο accuaracy. Βέβαια το δίκτυο δεν πρέπει να είναι
-->

Παρακάτω φαίνεται ενδεικτικά ο κώδικας που παρήγαγε αυτά τα αποτελέσματα και βρίσκεται στο `main.py`:

```{.python}
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
```

Όπως βλέπουμε ορίζουμε μία λίστα με όνομα `configs` που περιέχει tuples της μορφής `(layers, epochs)`.
Μετά διατρέχουμε αυτήν την λίστα με ένα for loop και κάνουμε fit το νευρωνικό δίκτυο με το ανάλογο πλήθος επιπέδων, πλήθος νευρώνων ανά επίπεδο και εποχές εκπαίδευσης. Τέλος απλά υπολογίζουμε το 
accuracy καλώντας την score και τυπώνουμε το αποτέλεσμα στην έξοδο.


### Ερώτημα (γ)

Στο βήμα αυτό χωρίσαμε το dataset σε train και validation sets κατά $80\%$
και $20\%$ αντίστοιχα. Το validation set το χρησιμοποιούμε στο τέλος κάθε εποχής
για τον υπολογισμό του accuracy πάνω σε αυτό, ώστε να μπορούμε να παρακολουθούμε την
πορεία της εκπαίδευσης. Επίσης στο τέλος κάθε εποχής υπολογίζουμε το accuracy πάνω στο train set. Για να υπολογίσουμε τα accuracies φτιάξαμε μια βοηθητική
συνάρτηση, την `eval_on_dataset`, που επέστρεφε τις προβλέψεις του τρέχοντος μοντέλου και μετά χρησιμοποιούσαμε την έτοιμη συνάρτηση `accuracy_score` από την sklearn.

 Χρησιμοποιήσαμε ως συνάρτηση κόστους το cross entropy και πιο συγκεκριμένα την κλάση `CrossEntropyLoss` και ως optimizer τον αλγόριθμο Adam. Το πλήθος
των εποχών είναι προκαθορισμένο κατά την δημιουργία του `PytorchNNModel`.

Πέρα από αυτά δεν κάναμε κάτι ιδιαίτερο για την διαδικασία
της εκπαίδευσης, απλά φτιάξαμε δυο εμφολιασμένα for loop. Το πρώτο αφορά τις
εποχές ενώ το δεύτερο απλά φορτώνει όλα τα δεδομένα του train set με την χρήση
του αντίστοιχου dataloader. Παρακάτω φαίνεται συνολικά η συνάρτηση fit():

```{.python}
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
```

Για την predict χρησιμοποιούμε έναν dataloader πάνω στα δεδομένα που μας δίνονται
και τα διατρέχουμε με ένα for loop. Σε κάθε επανάληψη του for loop το δίκτυο
δέχεται 16 samples στην είσοδο του και λαμβάνουμε τις 16 αντίστοιχες εξόδους.
(Οι έξοδοι αποτελούν logits και δεν έχουμε περάσει από την softmax.) Για να 
ταξινομήσουμε μετά τα δείγματα απλά παίρνουμε το argmax πάνω στα logits, δηλαδή
τα ταξινομούμε στην κλάση με το μεγαλύτερο logit. (Αυτό είναι ισοδύναμο με το
να εφαρμόζαμε πρώτα softmax και μετά να παίρναμε το argmax.) Ενδεικτικά παρακάτω
φαίνεται ο κώδικας για την predict


```{.python}
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
```

Στην συνάρτηση score, απλά καλούμε την predict και χρησιμοποιούμε την συνάρτηση
`accuracy_score` της sklearn.

### Ερώτημα (δ)

Το σχολιασμό τον κάναμε στο βήμα (β) [εδώ](#nn-acc_commnets)

<!-- τίποτα παραπάνω για εδώ???-->
