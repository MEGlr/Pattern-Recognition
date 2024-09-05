# Τι περιέχει αυτός ο φάκελος;

Αυτός ο φάκελος περιέχει τον κώδικα που αφορά
την εκπαίδευση των lstm. Ο κώδικας ακολουθεί
την δομή του βοηθητικού κώδικα που μας δίνεται
αφού συμπληρώσαμε σε αυτόν τα κενά που υπήρχαν.
Πέρα από αυτά φτιάξαμε και μερικά δικά μας script.

Συνολικά τα αρχεία είναι τα εξής:


* `prelab_part1.py` : Αυτό το script υπολογίζει ότι ζητείται από τα βήματα 1 έως 4 της προπαρασκευής.
* `convolution.py` 
* `dataset.py`
* `evaluate.py`: για το evaluation των αποθηκευμένων μοντέλων
* `feature_extractor.py`
* `hyper_lstm.py` : script που χρησιμοποιείται για το hyperparameter tuning των lstm
* `lstm.py`
* `train.py`: εδώ ορίζονται συναρτήσεις για την εκπαίδευση
* `main_train.py` : script που χρησιμοποιείται για την εκπαίδευση
* `make_hyper_plots.py` : script για την δημιουργία των γραφικών που αφορούν το hyper parameter tuning των lstm
* `modules.py`


# Πως τρέχουμε τα script (Για τα βήματα 5 και 6 κατά κύριο λόγο)

## `hyper_lstm.py`

```
$ python hyper_lstm.py dataset_type 
```

όπου το `dataset_type` προσδιορίζει το είδος του dataset
που θα χρησιμοποιηθεί. Το `dataset_type` παίρνει τις τιμές:

| dataset_type          | Περιγραφή                                     |
| ---:                  |  :---                                         |
| spectrogram           | φασματογραφήματα                              |
| chroma                | χρωμογραφήματα                                |
| fused                 | φασματογραφήματα + χρωμογραφήματα             |
| spectrogram_beat      | Beat synced φασματογραφήματα                  |
| chroma_beat           | Beat synced χρωμογραφήματα                    |
| fused_beat            | Beat synced φασματογραφήματα + χρωμογραφήματα |

## `main_train.py`

```
$ python main_train.py dataset_type max_epochs [best]")
```

με

* `dataset_type` να προσδιορίζει το είδος του dataset
* `max_epochs` να προσδιορίζει το μέγιστο πλήθος από εποχές
* `best`: Αν θέλουμε να χρησιμοποιηθούν οι καλύτεροι υπερπαράμετροι που βρέθηκαν από το hyperparameter tuning τότε γράφουμε στο τέλος της εντολής την λέξη best. Αν δεν την γράψουμε τότε χρησιμοποιούνται οι default υπερπαράμετροι.

## `make_hyper_plots.py`

```
$ python make_hyper_plots.py
```

## `evaluate.py`

```
$ python dataset_type path_to_pickle
```

όπου το `dataset_type` παίρνει τις ίδιες τιμές όπως και στο `hyper_lstm.py` και προσδιορίζει
το είδος του dataset και το path to pickle είναι προφανώς το μονοπάτι προς το αποθηκευμένο μοντέλο.


