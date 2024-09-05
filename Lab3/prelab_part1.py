#import necessary libraries
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from help_code import *

from pathlib import Path


# help function for printing in console
def step_seperator(step_num):
    """
    Just prints a message to the stdout
    to inform the user which step is currently running
    """

    print("\n\033[1;32m[*] Running Step {} ...\033[0m\n".format(step_num))


# help-functions for reading and plotting spectrograms

def read_mel_spectrogram(spectrogram_file):
    spectrogram_file = Path(spectrogram_file) # work in any OS
    spectrogram = np.load(spectrogram_file)[:128]
    return spectrogram

def plot_spectrogram(mel, title):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")


# help-functions for reading and plotting chromagrams

def read_chromagram(spectrogram_file):
    chromagram_file = Path(spectrogram_file) # work in any OS
    chromagram = np.load(spectrogram_file)[128:]
    return chromagram


def plot_chroma(chroma, title):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax)
    

#################
# STEP 1 ########
#################

step_seperator(1)
p = Path("input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt")
f = open(p, "r")

# read filename - genre pairs

def extract_pair_from_str(input_line):
    filename, genre = input_line.split()
    filename = filename[:-3] # remove the .gz entension

    return (filename, genre)

pairs = f.readlines()[1:]
pairs = [extract_pair_from_str(x) for x in pairs]

# choose 2 FMA samples of different genre
selected_pairs = [pairs[0], pairs[160]]
print(f"Selected one {selected_pairs[0][1]} sample and one {selected_pairs[1][1]} sample.")



# read & plot spectrograms

selected_mel = []

for filename, genre in selected_pairs:
    train_folder = 'input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train'
    full_path = train_folder + "/" + filename
    mel = read_mel_spectrogram(full_path)
    selected_mel.append(mel)
    plot_spectrogram(mel, 'Spectrogram for ' + genre)
    plt.savefig(f'plots/spectrogram_{genre}.png', dpi=300)


#################
# STEP 2 ########
#################


step_seperator(2)


# a -------------------------------

# print spectrogram dimensions

print("Spectrogram dimensions : ")
print(selected_mel[0].shape,f"({selected_pairs[0][1]})")
print(selected_mel[1].shape, f"({selected_pairs[1][1]})")
print()

# b -------------------------------

# read & plot beat - synced spectrograms

selected_mel_beat = []

for filename, genre in selected_pairs:

    train_folder = 'input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train'
    full_path = train_folder + "/" + filename
    mel = read_mel_spectrogram(full_path)
    selected_mel_beat.append(mel)
    plot_spectrogram(mel, 'Beat Synced Spectrogram for ' + genre)
    plt.savefig(f'plots/beat_synced_spectrogram_{genre}.png', dpi=300)


# print their dimensions 

print("Beat-synced Spectrogram dimensions :")
print(selected_mel_beat[0].shape, f"({selected_pairs[0][1]})")
print(selected_mel_beat[1].shape, f"({selected_pairs[1][1]})")
print()


#################
# STEP 3 ########
#################


step_seperator(3)

# plot chromograms for the blues and chiptune samples we selected above 

selected_chroma = []

for filename, genre in selected_pairs:

    train_folder = 'input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train'
    full_path = train_folder + "/" + filename
    chroma = read_chromagram(full_path)
    selected_chroma.append(chroma)
    plot_chroma(chroma, 'Chromagram for ' + genre)
    plt.savefig(f'plots/chromagram_{genre}.png', dpi=300)

# print chromogram dimensions

print("Chromagram dimensions : ")
print(selected_chroma[0].shape,  f"({selected_pairs[0][1]})")
print(selected_chroma[1].shape, f"({selected_pairs[1][1]})")
print()


# plot beat synced chromograms for the blues and chiptune samples we selected above 

selected_chroma_beat = []

for filename, genre in selected_pairs:

    train_folder = 'input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train'
    full_path = train_folder + "/" + filename
    chroma = read_chromagram(full_path)
    selected_chroma_beat.append(chroma)
    plot_chroma(chroma, 'Beat Synced Chromagram for ' + genre)
    plt.savefig(f'plots/beat_synced_chromagram_{genre}.png', dpi=300)

print("Beat-synced Chromagram dimensions : ")
print(selected_chroma_beat[0].shape,  f"({selected_pairs[0][1]})")
print(selected_chroma_beat[1].shape,  f"({selected_pairs[1][1]})")
print()

#################
# STEP 4 ########
#################


step_seperator(4)

# define class mapping 
# format : 
#       initial genre - label : new label 
#                               None if we wish to ignore genre samples 

class_mapping = {
    'Rock': 'Rock',
    'Psych-Rock': 'Rock',
    'Indie-Rock': None,
    'Post-Rock': 'Rock',
    'Psych-Folk': 'Folk',
    'Folk': 'Folk',
    'Metal': 'Metal',
    'Punk': 'Metal',
    'Post-Punk': None,
    'Trip-Hop': 'Trip-Hop',
    'Pop': 'Pop',
    'Electronic': 'Electronic',
    'Hip-Hop': 'Hip-Hop',
    'Classical': 'Classical',
    'Blues': 'Blues',
    'Chiptune': 'Electronic',
    'Jazz': 'Jazz',
    'Soundtrack': None,
    'International': None,
    'Old-Time': None
}

# open train labels to calculate stats (count per genre before and after mapping)

# open .txt file containing labels 
labels = open('input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt', 'r')

# read file 
# first line = column names => ignore
lines = labels.readlines()[1:] 

# initialize a list for sample labels given in official dataset
initial_genres = []

# initialize a list for sample labels after applying class-mapping
processed_genres = []

for line in lines:
    # .txt format per sample (line): {.npy sample name}\t{genre-label}\n 
    # get label
    genre = line.split('\t')[1].strip('\n')

    # update initial label list
    initial_genres.append(genre)

    # map to new class with class_mapping dictionary and update processed_genres list
    processed_genres.append(class_mapping[genre])


# plot initial count per genre 
fig = plt.figure(figsize=(10, 8))
plt.tight_layout()
sns.histplot(data = initial_genres, shrink = 0.9)
plt.xticks(rotation=55)
plt.xlabel("Music Genre")
plt.ylabel("train samples count")
plt.title("Train samples per genre")
plt.savefig('plots/hist_initial_labels.png')

# plot count per genre after preprocessing
fig = plt.figure(figsize=(10, 8))
sns.histplot(data = processed_genres, shrink = 0.9)
plt.xticks(rotation=45)
plt.xlabel("Music Genre")
plt.ylabel("train samples count")
plt.title("Train samples per genre after preprocessing")
plt.savefig('plots/hist_processed_labels.png')
