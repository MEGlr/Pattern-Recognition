from convolution import CNNBackbone
from lstm import LSTMBackbone
from modules import Classifier, MultitaskRegressor, Regressor
import torch

from tqdm import tqdm


def evaluate(model,test_dataloader, is_regression=False, device="cuda"):
    """
    This function get a model and a dataloader and returns the
    ground truth values from the dataloader and the model predictions.
    """


    model.eval() #evaluation mode
    
    # labels_for_batch[i] = pytorch tensor that contains the labels
    # for each sample in the minibatch i
    # the same is true for outputs_for_batch
    labels_for_batch = []
    outputs_for_batch = []
    
    with torch.no_grad(): # don't compute gradients
        for inputs, labels, lengths in tqdm(test_dataloader):

            # mode data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # evaluate 
            _, pred = model(inputs, labels, lengths)

            # if it is not a regression problem then
            # return the class with the highest logit
            if (not is_regression):
                pred = torch.argmax(pred, dim = 1)

            # save the labels and the logits for this batch
            labels_for_batch.append(labels)
            outputs_for_batch.append(pred)
            
    # concatenate the results from each batch
    labels, preds = torch.cat(labels_for_batch), torch.cat(outputs_for_batch)

    # return the predictions and the ground truth values
    return preds.tolist(), labels.tolist()  # Return the model predictions + labels 


def kaggle_submission(model, test_dataloader, device="cuda"):
    outputs = evaluate(model, test_dataloader, device=device)
    # TODO: Write a csv file for your kaggle submmission
    raise NotImplementedError("You need to implement this")


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import sys
    #import pickle
    from dataset import *
    from pathlib import Path


    # ========================= Look up Tables ====================

    # LUT for read function based on the dataset_type
    READ_FN = {
        'spectrogram':      read_mel_spectrogram,
        'chroma':           read_chromagram,
        'fused':            read_fused_spectrogram,
        'spectrogram_beat': read_mel_spectrogram,
        'chroma_beat':      read_chromagram,
        'fused_beat':       read_fused_spectrogram 
    }

    # LUT for the path of the dataset based on the dataset_type
    DATASET_PATH = {
        'spectrogram': "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms",
        'chroma': "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms",
        'fused': "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms",

        'spectrogram_beat': "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat",
        'chroma_beat': "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat",
        'fused_beat': "../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat",
    }

    # ========================= Get cli params ====================

    if(len(sys.argv) != 3):
        print("Usage: python evaluate.py {dataset_type} {path_to_pickle}")
        print("where dataset_type in", list(READ_FN.keys()))
        quit()

    # load params
    dataset_type = sys.argv[1]
    saved_model_path = sys.argv[2]
    print(saved_model_path)

    # ======================== Load the Model =====================

    model = torch.load(saved_model_path, map_location=torch.device('cpu'))

    print(f"\n\nEvaluting on {DATASET_PATH[dataset_type]}\n\n")

    # ============================ load the Dataset =================

    test_dataset = SpectrogramDataset(
        DATASET_PATH[dataset_type],
        train=False,
        class_mapping=CLASS_MAPPING,
        max_length=1,
        read_spec_fn=READ_FN[dataset_type])

    test_loader = DataLoader(test_dataset,
                          batch_size=4)

    # ============================ Evaluate the model ==============
    
    y_pred, y_true = evaluate(model, test_loader, device="cpu")


    # ===================== Confusion matrix ======================

    target_names = test_dataset.label_transformer.inverse(range(10))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    # misc things for the plot
    disp.plot()
    plt.xticks(rotation = 90)
    plt.tight_layout()

    # save the plot
    saved_model_path = saved_model_path.replace('\\', '/') # in case it is run in windows
    name = saved_model_path.split('/')[-2]; 
    fpath = Path(f'../plots/cm_{name}.png')
    plt.savefig(fpath, dpi=300)

    # ================== Get the Classification Report =================

    # classifier report
    print(classification_report(y_true, y_pred, zero_division=0, target_names=target_names))
