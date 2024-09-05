import copy
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

# HINT: Use this class mapping to merge similar classes and ignore classes that do not work very well
CLASS_MAPPING = {
    "Rock": "Rock",
    "Psych-Rock": "Rock",
    "Indie-Rock": None,
    "Post-Rock": "Rock",
    "Psych-Folk": "Folk",
    "Folk": "Folk",
    "Metal": "Metal",
    "Punk": "Metal",
    "Post-Punk": None,
    "Trip-Hop": "Trip-Hop",
    "Pop": "Pop",
    "Electronic": "Electronic",
    "Hip-Hop": "Hip-Hop",
    "Classical": "Classical",
    "Blues": "Blues",
    "Chiptune": "Electronic",
    "Jazz": "Jazz",
    "Soundtrack": None,
    "International": None,
    "Old-Time": None,
}


def torch_train_val_split(
        dataset, batch_train, batch_eval,
        val_size=.2, shuffle=True, seed=None):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))

    # The indices list contains the index of each
    # sample of the dataset. We use it to split
    # the dataset into a train and val set.


    if shuffle:
        # suffle the indices
        np.random.seed(seed)
        np.random.shuffle(indices)

    # keep the first val_split shuffled elements
    # for the validation set and use th rest for
    # the train set
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,
                              batch_size=batch_train,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset,
                            batch_size=batch_eval,
                            sampler=val_sampler)

    return train_loader, val_loader


# Helper functions to read fused, mel, and chromagram
def read_fused_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    return np.float32(spectrogram.T)


def read_mel_spectrogram(spectrogram_file):
    #print("reading ", spectrogram_file)
    spectrogram = np.load(spectrogram_file)[:128]
    return np.float32(spectrogram.T)

    
def read_chromagram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[128:]
    return np.float32(spectrogram.T)


class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[: self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1


class SpectrogramDataset(Dataset):


    def __init__(self, path, class_mapping=None, train=True, max_length=-1, regression=None, read_spec_fn=read_fused_spectrogram):
    
        # set the paths
        self.train = train
        t = "train" if train else "test"
        p = os.path.join(path, t)


        self.regression = regression
        
        # ----------------------------------------------------------------------------------
        # TODO: this has been added to the original - needed for padding as max_length in train set is different from the one in test set 
        # need padding for the linear layer in CNN 

        if max_length > 0 and not regression:
            # specify max_length if we are not given a negative value 
            # must find max feat len of all train & test set data
            temp_t = "train" if not  train else "test"
            temp_p = os.path.join(path, temp_t)
            temp_index = os.path.join(path, "{}_labels.txt".format(temp_t))
            temp_files, _ = self.get_files_labels(temp_index, class_mapping)
            temp_feats = [read_spec_fn(os.path.join(temp_p, f)) for f in temp_files]
            temp_lengths = [len(i) for i in temp_feats]
            max_length = max(temp_lengths)
        # ---------------------------------------------------------------------------------

        # load the index file and get files-label pairs
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        
        # actually load the data
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        
        # get the dimension of the feature vector (the frequency axis)
        self.feat_dim = self.feats[0].shape[1]
        
        # get the lengths of each sequence, the max length
        # and create a corresponding PaddingTransform
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max(max_length,max(self.lengths)) # TODO: originally : ... else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        
        # transform the labels to numeric values if needed
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            if not regression:
                self.labels = np.array(
                    self.label_transformer.fit_transform(labels)
                ).astype("int64")
            else:
                self.labels = np.array(labels).astype("float64")


    def get_files_labels(self, txt, class_mapping):

        # read the file
        with open(txt, "r") as fd:
            lines = [l.rstrip().split("\t") for l in fd.readlines()[1:]]

        files, labels = [], []

        for l in lines:

            # get the correct column for the specified regression task
            # regression = 1 --> valence
            # regression = 2 --> energy
            # regression = 3 --> danceability
            if self.regression:
                l = l[0].split(",")
                files.append(l[0] + ".fused.full.npy")
                if (self.regression == ":"):
                    labels.append(l[:])
                else:
                    labels.append(l[self.regression])
                continue

            # if classification
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]

            # ignore it if mapped to None
            if not label:
                continue

            fname = l[0]
            if fname.endswith(".gz"):
                fname = ".".join(fname.split(".")[:-1])

            # handle beatsync filenames
            # e.g. 1042.beatsync.fused.npy -> 1042.fused.full.npy
            fname = fname.replace(".beatsync.fused", ".fused.full")

            #if(self.train == False):
            number = fname.split(".")[0]
            fname = number + ".fused.full.npy"
            
            files.append(fname)
            labels.append(label)
        return files, labels


    def __getitem__(self, item):
        # TODO: Inspect output and comment on how the output is formatted

        # the output is a tuple containing:
        # 1) the sequence of feature vectors which are truncated or zero padded
        #       depending on the self.max_length
        # 2) The label of the sequence. It is just a number becuase we used the LabelTransformer
        # 3) the length of the sequnce without the zero padding

        length = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], length
   

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    dataset = SpectrogramDataset(
        "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms",
        class_mapping=CLASS_MAPPING,
        train=True,
        read_spec_fn=read_mel_spectrogram
    )

    print(dataset[10])
    print(f"Input: {dataset[10][0].shape}")
    print(f"Label: {dataset[10][1]}")
    print(f"Original length: {dataset[10][2]}")
